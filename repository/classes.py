from translation import translate_language_async
from utils import split_text
from query import get_api_output, get_local_model_output
from deep_translator import GoogleTranslator
from model import load_bloomz
from openai import OpenAI
from utils import clean_str
from eval import plot, plot_de
import google.generativeai as genai
import pandas as pd
import json
import csv
import os 
from logger import logger
import asyncio

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
    "a#: a/b/c/d \n\n"
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
            self.output[language] is None or self.output[language] == 'null'):
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

    def _assign(self, lang, idx, question, answer, output, score):
        """Assign contents to Sample"""
        self.idx = idx
        self.questions[lang] = question
        self.answers[lang] = answer
        self.output[lang] = output
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
            score = data.get(f"score_{language}", None)
            sample._assign(language, idx, question, answer, output, score)
        
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
            asyncio.run(self.translate_async(target=language, mode=mode))
            # self.translate(target=language, mode=mode)
        if save:
            self._save_translated_benchmark()

    def run(self, config, print_results=True, plot_results=False):
        """Main experiment function"""
        if not self.languages or not self.samples:
            raise ValueError("Please ensure this MultilingualBenchmark contains samples and target languages!")
        
        # Load translated benchmark already if it exists 
        translated_path = f'repository/benchmarks/translated/{self.benchmark_name}_{len(self.samples)}.json'
        if not os.path.exists(translated_path):
            self.translate_all(mode='q', save=True)
        else:
            new_instance = self.from_json(config)
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
            self._plot_results()

        logger.info('Done running!')
        self.write_to_json(f'{self.config['experiments']['results_path']}/{self.benchmark_name}_{self.model_name}.json')
    
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
        print("MultilingualBenchmark.translate() STILL USED!") # debug
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

    async def translate_async(self, target, mode='q'):
        """
        Main function to translate question (mode = 'q') or question-output (mode = 'qa') asynchronously
        """
        logger.info(f"Translating {'questions' if mode=='q' else 'output'} from {'en' if mode=='q' else target} to {target if mode == 'q' else 'en'}...")
        
        if mode == 'q':
            translator = GoogleTranslator(target=target)
        elif mode == 'qo':
            translator = GoogleTranslator(target='en')

        # Translate all samples at once for the target language asynchronously
        translated_samples = await translate_language_async(translator, target, self.samples, delay=5, mode=mode)

        # Assign translations to sample attributes
        try:
            for i, text in enumerate(translated_samples):
                try:
                    if mode == 'q':
                        translated_question = text
                        self.samples[i]._add_to_language(target, 
                                                        mode=mode, 
                                                        to_add=translated_question)
                    elif mode == 'qo':
                        splits = split_text(text, delimiter=self.delimiter)
                        translated_output = splits[1]
                        self.samples[i]._add_to_language(target, 
                                                        mode=mode, 
                                                        to_add=translated_output)
                except:
                    logger.warning(f"Splitting failed for text {text}.. Adding None")
                    self.samples[i]._add_to_language(target, type=mode, to_add=None)

            logger.info("Translation successful!")
        except:
            logger.error('Translating samples failed, leaving translations as None instead.')



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
                answer = output[sample.idx]
                sample.output[language] = answer

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

    def evaluate_all(self):
        """Evaluate for all languages their answers using direct answer matching"""
        logger.info("Evaluating")
        for sample in self.samples:
            sample._evaluate()

    def _get_valid_evals(self, language):
        evals = [sample.evaluations.get(language, None) for sample in self.samples]
        return [float(eval) for eval in evals if (eval != 'null') and (eval is not None)]
    
    def _valid_evals_to_accuracy(self, valid_evals):
        # Calculate accuracy by excluding -1 responses
        valid_scores = [score for score in valid_evals if score != -1]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0

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

            output += f"Target Language: {language}\n"
            output += f"a: {self.metrics[language]['a']}\n"
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

            self.metrics[language].update({'a': acc})

        if save:
            self._save_metrics()

    def _save_metrics(self):
        # Define the path to the CSV file
        path = 'research/metrics.csv'
        existing = []

        # Check if the file already exists
        file_exists = os.path.exists(path)

        # If the file exists, load the existing metrics
        if file_exists:
            with open(path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.append(row)
            logger.info(f"{path} exists. Loaded existing metrics.")
        else:
            logger.info(f"{path} does not exist. Will create a new metrics file.")

        # Open the file in append mode to add new rows without overwriting existing data
        with open(path, mode='a', newline='') as f:
            columns = ['model', 'benchmark', 'lang', 'a']  # Ensure all columns are included
            writer = csv.DictWriter(f, fieldnames=columns)

            # If the file is being created, write the header
            if not file_exists:
                writer.writeheader()

            # Write new rows if they do not yet exist
            for lang, values in self.metrics.items():
                row = {'model': self.model_name,
                    'benchmark': self.benchmark_name,
                    'lang': lang,
                    **values}
                if row not in existing:
                    writer.writerow(row)
                    logger.info(f"Added new row for lang {lang}.")
                else:
                    logger.warning(f"Duplicate row found for lang {lang}: skipping.")

        logger.info("Successfully saved metrics.")

    def _save_translated_benchmark(self, path='repository/benchmarks/translated/'):
        """Save translated QA pairs as .json - saves a lot of time during the experiment"""
        save_path = f'{path}{self.benchmark_name}_{len(self.samples)}.json' 
        logger.info(f"Now saving translated benchmark to {save_path}")
        self.write_to_json(save_path)
        logger.info("Saved succesfully!")
            
    def _plot_results(self):
        if self.has_results:
            # Read CSV with metrics
            metrics = pd.read_csv(f'research/metrics.csv').dropna()

            # Plot results
            plot(metrics)
            plot_de(metrics)
        else:
            logger.info('No results recorded yet for plotting. Please run cls.run() first.')


    @classmethod
    def from_json(cls, config):
        """Initialize class from configuration"""
        # Read arguments, joint for conciseness and separately for logging and path referencing
        args = read_args(config)
        benchmark, model, run, languages = args
        logger.info(f'Loading {benchmark} for {run} with {model} from json...')

        # Initialize Benchmark object from JSON
        benchmark = cls(args)
        with open(f"{config['paths']['results']}\{model}_{benchmark}", 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Assign file contents to this benchmark's samples
        for i, item in enumerate(data):
            sample = Sample.from_dict(item, idx=i, languages=languages)
            benchmark.add_sample(sample)

        # Set results toggle to True
        benchmark.has_results = True

        return benchmark
        
# TODO: Move to utils
def create_samples(json_object: dict) -> list:
    return [Sample(pair['question_en'], pair['answer_en'], pair['idx']) for pair in json_object]

def read_args(config: dict):
    """Reads arguments required for benchmark initialization from configuration"""
    benchmark=config['benchmark']['name'],
    model=config['benchmark']['model_name'],
    run=config['benchmark']['run_name'],
    languages=config["languages_subsets"][config["experiments"]["language_subset"]]["iso_639_1"],
    config=config
    return benchmark, model, run, languages, config

def run_experiments(benchmark, config):
    """Run the experiments and save the results."""
    logger.info("Now running experiments.")
    
    # Load data, add to benchmark object
    data = get_subset(config['benchmark'].get('path'), n=config["benchmark"]["subset_size"])
    benchmark.load_benchmark(data)
    
    benchmark.run(print_results=True, plot_results=True)
    
    benchmark.write_to_json(
        f'repository/benchmarks/results/{benchmark.benchmark_name}_{benchmark.model_name}.json'
    )

def load_previous_results(benchmark, config):
    """Load previous experiment results from a JSON file."""
    previous_benchmark = benchmark.from_json(config)
    
    previous_benchmark._get_metrics()
    previous_benchmark.print_results()
    previous_benchmark._plot_results()