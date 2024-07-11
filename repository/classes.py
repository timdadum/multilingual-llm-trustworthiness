from translation import translate_batch
from utils import split_text
from query import get_gpt_output, get_local_model_output
from deep_translator import GoogleTranslator
from model import load_bloomz
from openai import OpenAI
from eval import plot_accuracy_vs_sim, plot_accuracy_vs_percentage, plot_surface_accuracy_vs_sim_and_percentage
import pandas as pd
import json
import os 
from logger import logger

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
        self.backtranslated_output = {}
        self.evaluations = {}

        # idx to keep track of sample position
        self.idx = idx

    def __str__(self, language='en'):
        """Represents object as string - language shows target language for this representation"""

        separator = "|"
        output_str = f"q{self.idx}: {self.questions[language]} {separator} a{self.idx}: {self.answers[language]}"

        # Include output (if not None)
        output_str += f" {separator} o{self.idx}: {self.output.get(language, None)}"

        # Include output (if not None)
        output_str += f" {separator} bo{self.idx}: {self.backtranslated_output.get(language, None)}"

        # Include score (if not None)
        output_str += f" {separator} e{self.idx}: {self.evaluations.get(language, None)}"

        return output_str
    
    def _contains_nan(self, language):
        
        if (self.questions[language] is None or self.questions[language] == 'null' or
            self.answers['en'] is None or self.answers['en'] == 'null' or
            self.output[language] is None or self.output[language] == 'null' or
            self.backtranslated_output[language] is None or self.backtranslated_output[language] == 'null'):
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
     
    def _to_backtranslated_output_str(self, language='en'):
        return self.backtranslated_output.get(language, None)
       
    def _to_dict(self):
        """Returns this object as a dictionary (useful for writing to .json)"""
        result = {'idx': self.idx}
        for lang in self.questions:
            result[f"answer_en"] = self.answers['en']
            result[f"question_{lang}"] = self.questions.get(lang, None)
            result[f"output_{lang}"] = self.output.get(lang, None)
            result[f"backtranslated_output_{lang}"] = self.backtranslated_output.get(lang, None)
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

    def _add_to_language(self, language, mode='q', to_add=None, **kwargs):
        """
        Add to this sample's contents new content based on type.
        
        Args:
            language (str): Language to add to sample conform deep_translator.GoogleTranslator docs, e.g. 'cy' for Welsh, 'hi' for Hindi
            mode (str): 
            - 'q' for translation of questions, 
            - 'qo' for translation of questions to target and the output given in the target language (backtranslation)
            to_add (Any): Content to add
        Returns:
            A sample object with arguments as contents
        """
        if mode == 'q':
            self.questions[language] = to_add
        elif mode == 'qo':
            self.backtranslated_output[language] = to_add

    def _assign(self, lang, idx, question, answer, output, backtranslated_output, score):
        """Assign contents to Sample"""
        self.idx = idx
        self.questions[lang] = question
        self.answers[lang] = answer
        self.output[lang] = output
        self.backtranslated_output[lang] = backtranslated_output
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
            backtranslated_output = data.get(f"backtranslated_output_{language}", None)
            score = data.get(f"score_{language}", None)
            sample._assign(language, idx, question, answer, output, backtranslated_output, score)
        
        return sample



class MultilingualBenchmark:
    """
    Represents a benchmark consisting of multiple samples.

    Attributes
    ----------
    samples: list, a list of Sample objects
    current_idx: int,  
    """
    def __init__(self, model_name: str, run_name: str, languages: list, config: dict):
        self.samples = []
        self.current_idx = -1
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
            self.to_json(f'repository/benchmarks/translated/{self.run_name}_GoogleTranslated.json')

    def run(self, print_results=True, plot_results=False):
        """Main experiment function"""
        if not self.languages or not self.samples:
            raise ValueError("Please ensure this MultilingualBenchmark contains samples and target languages!")
        
        # Load translated benchmark already if it exists
        translated_path = f'repository/benchmarks/translated/{self.run_name}_GoogleTranslated.json'
        if not os.path.exists(translated_path):
            self.translate_all(mode='q', save=True)
        else:
            new_instance = self.from_json(translated_path, model=self.model_name, name=self.run_name, languages=self.languages, config=self.config)
            self.model_name = new_instance.model_name
            self.run_name = new_instance.run_name
            self.languages = new_instance.languages
            self.samples = new_instance.samples

        if 'gpt' in self.model_name.lower():
            self.query_gpt()
        elif 'bloomz' in self.model_name.lower():
            self.tokenizer, self.model = load_bloomz(config=self.config)
            self.query_bloomz()
        
        # Backtranslate outputs to English
        self.translate_all(mode='qo', save=False)

        # Evaluate targets and then filter for null values (TODO: Skip evaluating for any None-values)
        self.evaluate()
        self.has_results = True

        if print_results:
            self.print_results()

        if plot_results:
            self.plot_results()

        logger.info('Done running!')
    
    def __iter__(self):
        self.current_idx = -1
        return self
    
    def _json_format(self):
        return [sample._to_dict() for sample in self.samples]

    def to_json(self, path):
        logger.info("Writing benchmark to json...")
        
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self._json_format(), file, indent=4, ensure_ascii=False)
        
        logger.info(f"Benchmark {f'{self.run_name}_{self.model_name}'} uccesfully written to {path}")

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
    
    def _assign_output(self, target, output: dict):
        """
        Takes dict: {0: Mars, 1: five, 2: etc.}, where 0 represents sample with idx 0 and Mars represents the output
        """
        # Sort benchmark such that we can
        for sample in self.samples:
            if sample.idx in output:
                sample.output[target] = output[sample.idx]
            else:
                sample.output[target] = None

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

    
    def query_gpt(self, mode='ANS'):
        """
        Queries benchmark using all languages using GPT models
        """
        client = OpenAI()
        for language in self.languages:
            logger.info(f'Now querying {self.model_name} for language {language}')
            try:
                output = get_gpt_output(self, self.model_name, client, language, mode=mode, batch_size=self.gpt_batch_size)
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

    def _get_valid_evals(self, language):
        evals = [sample.evaluations[language] for sample in self.samples]
        return [float(eval) for eval in evals if eval is not None]
    
    def _valid_evals_to_accuracy(self, valid_evals):
        return sum(valid_evals) / len(valid_evals)

    def _extract_accuracies(self, thresh=8):
        """
        Converts experiment results to a dictionary of accuracies.

        Args:
            data (list of dictionaries): Object containing experiment results in standard format
            languages (list of str): List of languages in data
        Returns:
            scores (dict): A dictionary with per-language accuracy
        """
        scores = {}
        for language in self.languages:
            # Filter out null values and count the correct evaluations
            valid = self._get_valid_evals(language)
            if len(valid) > thresh:
                acc = self._valid_evals_to_accuracy(valid)
                scores[language] = acc
            else:
                logger.warning(f"Too few valid evaluations found ({len(valid)}, should be at least {thresh}) for language {language}. Outputting None.")
                scores[language] = None
        return scores
    
    def _save_translated_benchmark(self, translated_by='GTranslator', path='repository/benchmarks/translated/'):
        """Save translated QA pairs as .json - saves a lot of time during the experiment"""
        save_path = path + translated_by
        logger.info(f"Now saving translated benchmark to {save_path}")
        self.to_json(save_path)
        logger.info("Saved succesfully!")

    def print_results(self, return_scores=False):
        scores = self._extract_accuracies()

        # Formatting the output
        output = "\nExperiment Results:\n"
        output += "=" * 20 + "\n"
        for language, accuracy in scores.items():
            if accuracy is not None:
                output += f"Target Language: {language}\n"
                output += f"Accuracy: {accuracy * 100:.2f}%\n"
            else:
                output += f"Target Language: {language}\n"
                output += "Accuracy: None\n"
            output += "-" * 20 + "\n"

        print(output)

        if return_scores:
            return scores
        
    def plot_results(self, lang_iso=None, lang_trans=None, features_path='repository/features/language_features.csv'):
        if self.has_results:
            # Get langauge scores, transform keys to iso language codes to match feature language codes
            lang_iso = [
                'arb', 'fra', 'spa', 'hin',
                'zho', 'eng', 'cym', 'fin',
                'hun', 'zul', 'nld', 'ita',
                'vie', 'swh', 'jpn', 'deu',
                'ind', 'urd', 'rus', 'por',
                'ben'
            ]

            lang_trans = [
                'ar', 'fr', 'es', 'hi',
                'zh-CN', 'en', 'cy', 'fi',
                'hu', 'nl', 'it', 'bn',
                'vi', 'sw', 'ja', 'de',
                'id', 'ur', 'ru', 'pt',
            ]
           
            scores = self._extract_accuracies()
            iso_to_transformed = {iso: trans for iso, trans in zip(lang_iso, lang_trans)}
            transformed_to_iso = {value: key for key, value in iso_to_transformed.items()}
            iso_scores = {transformed_to_iso[lan]: score for lan, score in scores.items()}
            
            # Merge our results with predefined language features in a Pandas dataframe
            features = pd.read_csv(features_path)
            results = features.merge(pd.DataFrame(list(iso_scores.items()), columns=['language', 'accuracy']), on='language', how='left')
            results = results[results['language'] != 'eng']
            results = results.dropna()

            # Plot results
            plot_accuracy_vs_sim(results)
            plot_accuracy_vs_percentage(results)
            plot_surface_accuracy_vs_sim_and_percentage(results)
        else:
            logger.info('No results recorded yet for plotting. Please run cls.run() first.')


    @classmethod
    def from_json(cls, path, model, name, languages, config):
        logger.info(f'Loading {path} for {name}_{model} from json...')

        # Load Benchmark object
        benchmark = cls(model, name, languages, config)
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
