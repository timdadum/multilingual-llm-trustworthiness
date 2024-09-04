from translation import translate_batch
from utils import split_text
from query import get_api_output, get_local_model_output
from deep_translator import GoogleTranslator
from model import load_bloomz
from openai import OpenAI
from utils import clean_str, calculate_uncertainty_rate, calculate_uncertainty_accuracy, calculate_uncertainty_f1
from eval import plot_separate_metrics_per_language
import google.generativeai as genai
import pandas as pd
import json
import csv
import os 
from logger import logger

### SYSTEM PROMPTS AND MISC. ### #TODO: REFORMAT

ANS_PROMPT_2 = (
    "## Objective:\n"
    "You are a highly accurate multiple-choice question answerer. Your task is to provide answers strictly following the designated format.\n\n"
    "## Response Format:\n"
    "- Answer Key:\n"
    "  Each answer must be formatted as follows:\n"
    "  a#: a/b/c/d\n\n"
    "  - '#' represents the question number (e.g., a0 for the first question, a1 for the second, etc.).\n"
    "## Example Input and Output:\n"
    "- Input:\n"
    "  - q0: What is the capital of France? a: Berlin, b: Madrid, c: Paris, d: Rome\n"
    "  - q1: What is the atomic symbol for carbon? a: CB, b: C, c: Gb, d: Cr\n\n"
    "- Output:\n"
    "  - a0: c\n"
    "  - a1: b@\n\n"
    "## Instructions:\n"
    "1. For each question, provide your answer in the following format:\n"
    "   - a#: <answer>\n"
    "2. List multiple answers sequentially:\n"
    "   - a0: <answer>\n"
    "   - a1: <answer>\n"
    "   - a2: <answer>\n"
    "   - …\n"
    "3. Ensure all responses are in the exact format specified above."
)

ANS_PROMPT = (
    "You are a highly accurate multiple-choice question answerer. Your responses must strictly adhere to the following format:\n"
    "a#: a/b/c/d/a \n\n"
    "Where # is the question number (e.g., a0 for the first question, a1 for the second).\n\n"
    "For multiple questions, provide answers in the format:\n"
    "a0: <answer>\n"
    "a1: <answer>\n"
    "a2: <answer>\n"
    "...\n\n"
    "## EXAMPLE\n"
    "q0: What is the capital of France? a: Berlin, b: Madrid, c: Paris, d: Rome\n"
    "q1: What is the atomic symbol for carbon? a: CB, b: C, c: Gb, d: Cr\n\n"
    "## OUTPUT\n"
    "a0: c\n"
    "a1: b"
)

# Used for Google API safety blockage
safe = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    }
]

######################

class Sample:
    """
    Represents a sample. Contains dictionaries with per-language questions

    Attributes
    ----------
    questions: str, the questions in the sample
    answers: str, the answers for the sample
    output: str, the output generated for the sample
    scores: dict, dictionary where key is language (str) and value is score (int, binary)
    """
    def __init__(self, question, answer, idx):
        # Create attributes
        self.questions = {'en': question}
        self.answers = {'en': answer}
        self.output = {}
        self.evaluations = {}
        self.uncertainties = {}

        # idx to keep track of sample position
        self.idx = idx

    def __str__(self, language='en'):
        """Represents object as string - language shows target language for this representation"""

        separator = "|"
        output_str = f"q{self.idx}: {self.questions[language]} {separator} a{self.idx}: {self.answers[language]}"

        # Include output (if not None)
        output_str += f" {separator} o{self.idx}: {self.output.get(language, None)}"

        # Include score (if not None)
        output_str += f" {separator} e{self.idx}: {self.evaluations.get(language, None)}"

        return output_str
    
    def _contains_nan(self, language):
        
        if (self.questions[language] is None or self.questions[language] == 'null' or
            self.answers['en'] is None or self.answers['en'] == 'null' or
            self.output[language] is None or self.output[language] == 'null' or
            self.uncertainties[language] is None or self.uncertainties[language] == 'null'):
                return True
        else:
            return False

    # The above method but as separate functions 
    def _to_question_str(self, language='en'):
        return self.questions.get(language, None)

    def _to_answer_str(self, language='en'):
        return self.answers.get(language, None)
    
    def _to_output_str(self, language='en'):
        return self.output.get(language, None)
       
    def _to_dict(self):
        """Returns this object as a dictionary (useful for writing to .json)"""
        result = {'idx': self.idx}
        for lang in self.questions:
            result[f"answer_en"] = self.answers['en']
            result[f"question_{lang}"] = self.questions.get(lang, None)
            result[f"output_{lang}"] = self.output.get(lang, None)
            result[f"uncertainty_{lang}"] = self.uncertainties.get(lang, None)
            result[f"score_{lang}"] = self.evaluations.get(lang, None)
        return result

    @classmethod
    def from_string(cls, text, language):
        """
        Constructor class method, constructing from English 
        
        Args:
            text (str): Sample object contents, formatted as <QUESTION>, <ANSWER> (and optionally , <OUTPUT> , <SCORE>)
            language (str): Language to add to sample conform deep_translator.GoogleTranslator docs, e.g. 'cy' for Welsh, 'hi' for Hindi
        Returns:
            A sample object with arguments as contents
        """

        # Split input into sections
        parts = text.split(separator="|")

        # Infer QA pair as a minimum to create a Sample object
        try:
            question, answer = parts[0].strip(), parts[1].strip()
        except Exception as e:
            logger.warning(f"Error found in sample '{text}'. Returning None instead...")
            return None
        
        # Create object
        sample = cls(question, answer)

        # Potentially add output and score 
        if len(parts) >= 2:
            # There is an output in this string, add...
            output = parts[2].strip()
            sample.add_output(language, output)
        if len(parts) >= 3:
            # There is a score in this string, add...
            score = parts[3].strip()
            sample.score(language, score)

        return sample

    def _evaluate(self):
        """Evaluates using multiple-choice matching."""
        for language, output in self.output.items():
            try:
                out = clean_str(output)
            except (AttributeError, TypeError) as e:
                # Score should be null - invalid sample for this language
                self.score(language, 'null')
                continue
            
            ans = self.answers.get('en', '').lower().strip()

            # Checking if the cleaned output matches the answer or indicates "I do not know" (which is always F.)
            if out == ans:
                self.score(language, 1)
            elif out in ['a', 'b', 'c', 'd']:
                self.score(language, 0)
            else:
                self.score(language, 'null')

    def _get_question_uncertainty(self):
        return round(sum(self.uncertainties) / len(self.uncertainties), 2)

    def _add_to_language(self, language, mode='q', to_add=None, **kwargs):
        """
        Add to this sample's contents new content based on type.
        
        Args:
            language (str): Language to add to sample conform deep_translator.GoogleTranslator docs, e.g. 'cy' for Welsh, 'hi' for Hindi
            mode (str): 
            - 'q' for translation of questions, 
            - 'qo' for translation of questions to target and the output given in the target language
            to_add (Any): Content to add
        Returns:
            A sample object with arguments as contents
        """
        if mode == 'q':
            self.questions[language] = to_add
        elif mode == 'qo':
            self.output['en'] = to_add

    def _assign(self, lang, idx, question, answer, output, uncertain, score):
        """Assign contents to Sample"""
        self.idx = idx
        self.questions[lang] = question
        self.answers[lang] = answer
        self.output[lang] = output
        self.uncertainties[lang] = uncertain
        self.evaluations[lang] = score
        
    def add_output(self, language, output):
        self.output[language] = output

    def score(self, language, score):
        self.evaluations[language] = score

    @classmethod
    def from_dict(cls, data: dict, idx, languages):
        """"""
        question_en = data.get("question_en")
        answer_en = data.get("answer_en")
        sample = cls(question_en, answer_en, idx)

        # Assign anything in the dictionary to sample
        for language in languages:
            idx = data.get(f"idx")
            question = data.get(f"question_{language}", None)
            answer = data.get(f"answer_{language}", None)
            output = data.get(f"output_{language}", None)
            uncertain = data.get(f"uncertainty_{language}", None)
            score = data.get(f"score_{language}", None)
            sample._assign(language, idx, question, answer, output, uncertain, score)
        
        return sample


class MultilingualBenchmark:
    """
    Represents a benchmark consisting of multiple samples.

    Attributes
    ----------
    samples: list, a list of Sample objects
    current_idx: int,  
    """
    def __init__(self, benchmark_name: str, model_name: str, run_name: str, languages: list, config: dict):
        self.samples = []
        self.current_idx = -1

        # Names
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self.run_name = run_name

        # Configure
        self.config = config
        self.delimiter = config.get("misc").get("delimiter")
        self.translation_batch_size = config.get("misc").get("translation_batch_size")
        self.gpt_batch_size = config.get("misc").get("gpt_batch_size")

        # OPTIONAL: languages to experiment with. ensures this doesn't have to be 
        # a method argument all the time.
        self.languages = languages

        # Results
        self.metrics = {language: {} for language in languages}

        self.has_results = False

    def __str__(self):
        return f"Benchmark (model: {self.model_name}, experiment name: {self.run_name}, samples={len(self)})"
    
    def get_languages(self):
        print(f'Benchmark uses languages {self.languages}')

    def _sort(self):
        self.samples = sorted(self.samples, key=lambda obj: obj.idx)

    def add_sample(self, sample):
        """
        Adds a sample to the benchmark.

        Parameters
        ----------
        sample: Sample
            The Sample object to be added.
        """
        self.samples.append(sample)

    def load_benchmark(self, data: dict):
        """Uses predefined data format!"""
        for i, item in enumerate(data):
            sample = Sample(item['question_en'], item['answer_en'], idx=i)
            self.add_sample(sample)

    def translate_all(self, mode='q', save=True):
        for language in self.languages:
            self.translate(target=language, mode=mode)
        if save:
            self._save_translated_benchmark()

    def run(self, print_results=True, plot_results=False):
        """Main experiment function"""
        if not self.languages or not self.samples:
            raise ValueError("Please ensure this MultilingualBenchmark contains samples and target languages!")
        
        # Load translated benchmark already if it exists 
        translated_path = f'repository/benchmarks/translated/{self.benchmark_name}_{len(self.samples)}.json'
        if not os.path.exists(translated_path):
            self.translate_all(mode='q', save=True)
        else:
            new_instance = self.from_json(translated_path,
                                          benchmark_name=self.benchmark_name, 
                                          model_name=self.model_name, 
                                          run_name=self.run_name, 
                                          languages=self.languages, 
                                          config=self.config)
            self.model_name = new_instance.model_name
            self.run_name = new_instance.run_name
            self.languages = new_instance.languages
            self.samples = new_instance.samples

        if 'gpt' in self.model_name.lower():
            self.query_gpt()
        elif 'gemini' in self.model_name.lower():
            self.query_gemini()
        elif 'bloomz' in self.model_name.lower():
            self.tokenizer, self.model = load_bloomz(config=self.config)
            self.query_bloomz()
        
        # Evaluate targets and then filter for null values (TODO: Skip evaluating for any None-values)
        self.evaluate_all()
        self._get_metrics()
        self.has_results = True

        if print_results:
            self.print_results()

        if plot_results:
            self.plot_results()

        logger.info('Done running!')
        self.write_to_json(f'repository/benchmarks/results/{self.benchmark_name}_{self.model_name}.json')
    
    def __iter__(self):
        self.current_idx = -1
        return self
    
    def _json_format(self):
        return [sample._to_dict() for sample in self.samples]

    def write_to_json(self, path):
        logger.info("Writing benchmark to json...")
        
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self._json_format(), file, indent=4, ensure_ascii=False)
        
        logger.info(f"Benchmark {f'{self.run_name}_{self.model_name}'} succesfully written to {path}")

    def __next__(self):
        self.current_idx += 1
        if self.current_idx < len(self):
            return self.samples[self.current_idx]
        else:
            raise StopIteration

    def __len__(self):
        """
        Returns the number of samples in the benchmark.
        """
        return len(self.samples)
    
    def translate(self, target, mode='q'):
        """
        Main function to translate question (mode = 'q') or question-output (mode = 'qa')
        """
        logger.info(f"Translating {'questions' if mode=='q' else 'output'} from {'en' if mode=='q' else target} to {target if mode == 'q' else 'en'}...")
        
        # Create batches of samples
        batch_size = self.translation_batch_size
        batches = self.batchify(batch_size=batch_size)
        
        # Translate in batches
        if mode == 'q':
            translator = GoogleTranslator(target=target)
        elif mode == 'qo':
            translator = GoogleTranslator(target='en')

        translated_batches = [translate_batch(translator, target, batch, mode=mode) for batch in batches]

        # Assign idx-wise to sample attributes
        try:
            for i, batch in enumerate(translated_batches):
                for j, text in enumerate(batch):
                    try:
                        if mode == 'q':
                            translated_question = text
                            self.samples[i * batch_size + j]._add_to_language(target, 
                                                                                mode=mode, 
                                                                                to_add=translated_question)
                        elif mode == 'qo':
                            splits = split_text(text, delimiter=self.delimiter)
                            translated_output = splits[1]
                            self.samples[i * batch_size + j]._add_to_language(target, 
                                                                                mode=mode, 
                                                                                to_add=translated_output)
                    except:
                        logger.warning(f"Splitting failed for text {text}.. Adding None")
                        self.samples[i * batch_size + j]._add_to_language(target, type=mode, to_add=None)

            logger.info("Translation succesfull!")
        except:
            logger.error('Translating batch failed, leaving translations in batch as None instead.')

    def get_average_score(self, language):
        """
        Calculates the average score for a given language across all samples.

        Parameters
        ----------
        language: str
            The language for which the average score is to be calculated.

        Returns
        -------
        float
            The average score for the given language.
        """
        total_score = 0
        count = 0
        for sample in self.samples:
            if language in sample.scores:
                total_score += sample.scores[language]
                count += 1
        return total_score / count if count > 0 else 0

    def get_all_scores(self):
        """
        Gets all scores for all languages and samples.

        Returns
        -------
        dict
            A dictionary where keys are languages and values are lists of scores for each sample.
        """
        scores_dict = {}
        for sample in self.samples:
            for language, score in sample.scores.items():
                if language not in scores_dict:
                    scores_dict[language] = []
                scores_dict[language].append(score)
        return scores_dict
    
    def batchify(self, batch_size=16):
        batches = []
        start_idx = 0

        for _ in range(len(self) // batch_size):
            end_idx = start_idx + batch_size
            batch = self.samples[start_idx:end_idx]
            batches.append(batch)
            start_idx = end_idx
        
        # Handle leftover samples if any
        if len(self) % batch_size != 0:
            leftover_batch = self.samples[start_idx:]
            batches.append(leftover_batch)
        
        return batches
    
    def _assign_output(self, language, output: dict):
        """
        Takes dict: {0: Mars, 1: five, 2: etc.}, where 0 represents sample with idx 0 and Mars represents the output
        """
        # Sort benchmark such that we can
        for sample in self.samples: # TODO Substitute this with a non-looping solution
            if sample.idx in output:
                answer, uncertain = output[sample.idx]
                sample.output[language] = answer
                sample.uncertainties[language] = uncertain

    def _assign_evaluation(self, target: str, evals: dict):
        """
        Assign evaluations to samples.
        
        Parameters:
            target (str): The evaluation target.
            evals (dict): Dictionary where keys are sample indices and values are evaluation results.
                    Example: {0: "mars", 1: "five", 2: "etc."}
        """
        for sample in self.samples:
            contains_nan = sample._contains_nan(target)
            sample.evaluations[target] = None if contains_nan else evals.get(sample.idx, None)
    
    def query_gpt(self):
        """
        Queries benchmark using all languages using GPT models
        """
        client = OpenAI()
        for language in self.languages:
            logger.info(f'Now querying {self.model_name} for language {language}')
            try:
                output = get_api_output(self, language, api_type='openai', batch_size=self.gpt_batch_size, client=client, engine=self.model_name, sys_prompt=ANS_PROMPT)
                self._assign_output(language, output)
            except Exception as e:
                logger.critical(f"Error encountered querying {self.model_name} in language code {language}. Skipping language... \n ERROR: {e}")

    def query_gemini(self, engine='gemini-1.5-flash'):
        """
        Queries benchmark using all languages using Gemini models
        """
        for language in self.languages:
            # Translate system prompt
            sys_prompt = GoogleTranslator(target=language).translate(ANS_PROMPT)
            
            model = genai.GenerativeModel(
                model_name=engine,
                system_instruction=sys_prompt,
                safety_settings=safe
            )

            logger.info(f'Now querying {self.model_name} for language {language}')
            try:
                output = get_api_output(self, language, api_type='google', batch_size=self.gpt_batch_size, model=model)
                self._assign_output(language, output)
            except Exception as e:
                logger.critical(f"Error encountered querying {self.model_name} in language code {language}. Skipping language... \n ERROR: {e}")

    def query_bloomz(self, mode='ANS'):
        """
        Queries benchmark using all languages using BLOOMZ models
        """
        for language in self.languages:
            logger.info(f'Now querying {self.model_name} for language {language}')
            try:
                output = get_local_model_output(self, language, mode=mode)
                self._assign_output(language, output)
            except Exception as e:
                logger.critical(f"Error encountered querying {self.model_name} in language code {language}. Skipping language... \n ERROR: {e}")

    def evaluate(self):
        """
        UNUSED

        Evaluates benchmark using all languages using GPT models
        """
        client = OpenAI()
        for language in self.languages:
            logger.info(f'Now evaluating {self.model_name} for language {language}')
            try:
                evals = get_gpt_output(self, self.model_name, client, language, mode='EVAL')
                self._assign_evaluation(language, evals)
            except Exception as e:
                logger.error(f"Error encountered evaluating {self.model_name} in language code {language}. Skipping language... \n ERROR: {e}")

    def evaluate_all(self):
        """Evaluate for all languages their answers using direct answer matching"""
        logger.info("Evaluating")
        for sample in self.samples:
            sample._evaluate()

    def _get_valid_evals(self, language):
        evals = [sample.evaluations[language] for sample in self.samples]
        return [float(eval) for eval in evals if eval != 'null']
    
    def _valid_evals_to_accuracy(self, valid_evals):
        # Calculate accuracy by excluding -1 responses
        valid_scores = [score for score in valid_evals if score != -1]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0

    def _get_uncertainty_metrics(self, lang):
        # Create list of uncertainties and evaluations
        uncertainties = [u for sample in self.samples if (u := sample.uncertainties.get(lang)) is not None]
        evaluations = [e for sample in self.samples if (e := sample.evaluations.get(lang)) is not None]

        # Calculate true positives (TP), false positives (FP), and false negatives (FN)
        TP = sum(1 for u, e in zip(uncertainties, evaluations) if u and e == 0)
        FP = sum(1 for u, e in zip(uncertainties, evaluations) if u and e == 1)
        FN = sum(1 for u, e in zip(uncertainties, evaluations) if not u and e == 0)

        # Calculate metrics
        b = calculate_uncertainty_rate(uncertainties)
        y = calculate_uncertainty_accuracy(uncertainties, evaluations)
        k = calculate_uncertainty_f1(TP, FP, FN)

        return b, y, k

    def print_results(self):
        """Prints all results in an overview"""
        
        # Formatting the output
        output = "\nExperiment Results:\n"
        output += "=" * 20 + "\n"
        
        for language in self.metrics.keys():
            # Calculate valid samples
            valid_evals = self._get_valid_evals(language)
            valid_count = len(valid_evals)
            total_count = len(self.samples)
            
            a, b, y, k = (self.metrics[language].get(key) for key in ('a', 'b', 'y', 'k'))

            if all(metric is not None for metric in (a, b, y, k)):
                output += f"Target Language: {language}\n"
                output += f"a: {a * 100:.2f}%\n"
                output += f"b: {b * 100:.2f}%\n"
                output += f"y: {y * 100:.2f}%\n"
                output += f"k: {k * 100:.2f}%\n"
                output += f"Valid: {valid_count}/{total_count}\n"
            else:
                output += f"Target Language: {language}\n"
                output += "a: None\n"
                output += f"Valid: {valid_count}/{total_count}\n"
            output += "-" * 20 + "\n"

        print(output)

    def _get_metrics(self, save=True):
        """
        Converts experiment results to a dictionary of accuracies.

        Args:
            data (list of dictionaries): Object containing experiment results in standard format
            languages (list of str): List of languages in data
        Returns:
            scores (dict): A dictionary with per-language accuracy
        """
        for language in self.languages:
            valid = self._get_valid_evals(language)
            acc = self._valid_evals_to_accuracy(valid)
            b, y, k = self._get_uncertainty_metrics(language)

            self.metrics[language].update({'a': acc,
                                           'b': b,
                                           'y': y,
                                           'k': k})

        if save:
            self._save_metrics()

    def _save_metrics(self):
        path = f'research/{self.benchmark_name}_metric.csv'
        if os.path.exists(path):
            logger.info(f"{path} exists. Not saving metrics.")
        else:
            with open(path, mode='w', newline='') as f:
                columns = ['lang', 'a', 'b', 'y', 'k']
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for lang, values in self.metrics.items():
                    row = {'lang': lang, **values}
                    writer.writerow(row)
            logger.info("Succesfully saved metrics.")

    def _save_translated_benchmark(self, path='repository/benchmarks/translated/'):
        """Save translated QA pairs as .json - saves a lot of time during the experiment"""
        save_path = f'{path}{self.benchmark_name}_{len(self.samples)}.json' 
        logger.info(f"Now saving translated benchmark to {save_path}")
        self.write_to_json(save_path)
        logger.info("Saved succesfully!")
        
    def plot_results(self):
        if self.has_results:
            # Read CSV with metrics
            metrics = pd.read_csv(f'research/{self.benchmark_name}_metric.csv').dropna()

            # Plot results
            plot_separate_metrics_per_language(metrics)
        else:
            logger.info('No results recorded yet for plotting. Please run cls.run() first.')


    @classmethod
    def from_json(cls, path, benchmark_name, model_name, run_name, languages, config):
        logger.info(f'Loading {path} for {run_name}_{model_name} from json...')

        # Load Benchmark object
        benchmark = cls(benchmark_name, model_name, run_name, languages, config)
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Assign file contents to this benchmark's samples
        for i, item in enumerate(data):
            sample = Sample.from_dict(item, idx=i, languages=languages)
            benchmark.add_sample(sample)

        # Set results toggle to True
        benchmark.has_results = True

        return benchmark
        

def create_samples(json_object: dict) -> list:
    return [Sample(pair['question_en'], pair['answer_en'], pair['idx']) for pair in json_object]
