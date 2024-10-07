from translation import translate_batch, prepare_translation_batches
from query import get_api_output, get_local_model_output
from deep_translator import GoogleTranslator
from model import load_bloomz
from openai import OpenAI
from utils import clean_str, threadpool
from plot import plot, plot_de
import google.generativeai as genai
import pandas as pd
import json
import csv
import os 
from logger import logger

class Sample:
    """
    Represents a single sample of multiple-choice questions and answers.

    Attributes
    ----------
    questions : dict
        Dictionary where the key is the language code and the value is the question text.
    answers : dict
        Dictionary where the key is the language code and the value is the answer text.
    output : dict
        Dictionary where the key is the language code and the value is the model's output.
    evaluations : dict
        Dictionary where the key is the language code and the value is the evaluation evaluation.
    idx : int
        The index of the sample for tracking purposes.
    """
    def __init__(self, question, answer, idx):
        self.questions = {'en': question}
        self.answers = {'en': answer}
        self.output = {}
        self.evaluations = {}
        self.idx = idx

    def __str__(self, language='en'):
        """String representation of the sample object."""
        separator = "|"
        output_str = f"q{self.idx}: {self.questions[language]} {separator} a{self.idx}: {self.answers[language]}"
        output_str += f" {separator} o{self.idx}: {self.output.get(language, None)}"
        output_str += f" {separator} e{self.idx}: {self.evaluations.get(language, None)}"
        return output_str

    def _contains_nan(self, language):
        """Check if any key parts of the sample contain NaN or invalid values."""
        return (self.questions[language] in [None, 'null'] or
                self.answers['en'] in [None, 'null'] or
                self.output[language] in [None, 'null'])

    def _to_question_str(self, language='en'):
        """Return the question string for the given language."""
        return self.questions.get(language, None)

    def _to_answer_str(self, language='en'):
        """Return the answer string for the given language."""
        return self.answers.get(language, None)

    def _to_output_str(self, language='en'):
        """Return the output string for the given language."""
        return self.output.get(language, None)

    def _to_dict(self):
        """Convert the sample object into a dictionary."""
        result = {'idx': self.idx}
        for lang in self.questions:
            result[f"answer_en"] = self.answers['en']
            result[f"question_{lang}"] = self.questions.get(lang, None)
            result[f"output_{lang}"] = self.output.get(lang, None)
            result[f"evaluation_{lang}"] = self.evaluations.get(lang, None)
        return result

    @classmethod
    def from_string(cls, text, language):
        """Create a sample object from a formatted string."""
        parts = text.split(separator="|")
        try:
            question, answer = parts[0].strip(), parts[1].strip()
        except Exception as e:
            logger.warning(f"Error found in sample '{text}'. Returning None instead...")
            return None

        sample = cls(question, answer)
        if len(parts) >= 2:
            output = parts[2].strip()
            sample.add_output(language, output)
        if len(parts) >= 3:
            evaluation = parts[3].strip()
            sample.evaluation(language, evaluation)

        return sample

    def evaluate(self):
        """Evaluate the sample by comparing the output to the correct answer."""
        for language, output in self.output.items():
            try:
                out = clean_str(output)
            except (AttributeError, TypeError):
                self.evaluation(language, 'null')
                continue

            ans = self.answers.get('en', '').lower().strip()
            if out == ans:
                self.assign(1, language, data_type='evaluation')
            elif out in ['a', 'b', 'c', 'd']:
                self.assign(0, language, data_type='evaluation')
            else:
                self.assign('null', language, data_type='evaluation')

    def assign(self, value, language, data_type):
        """
        Adds a type of data to Sample dictionary.

        Parameters:
        -----------
        value : any
            The value to assign (e.g., a question, answer, output, or evaluation).
        language : str
            The language code to which the data belongs (e.g., 'en', 'fr').
        data_type : str
            The type of data being assigned (must be 'question', 'answer', 'output', or 'evaluation').
        """
        if data_type == 'question':
            self.questions[language] = value
        elif data_type == 'answer':
            self.answers[language] = value
        elif data_type == 'output':
            self.output[language] = value
        elif data_type == 'evaluation':
            self.evaluations[language] = value
        else:
            raise ValueError(f"Invalid data_type '{data_type}'. Must be 'question', 'answer', 'output', or 'evaluation'.")


    @classmethod
    def from_dict(cls, data, idx, languages):
        """Create a sample object from a dictionary."""
        question_en = data.get("question_en")
        answer_en = data.get("answer_en")
        sample = cls(question_en, answer_en, idx)

        for language in languages:
            question = data.get(f"question_{language}", None)
            answer = data.get(f"answer_{language}", None)
            output = data.get(f"output_{language}", None)
            evaluation = data.get(f"evaluation_{language}", None)
            
            # Assign all values
            for value in [question, answer, output, evaluation]:
                sample.assign(value, language, data_type=f"{str(value)}")

        return sample





class MultilingualBenchmark:
    """
    Represents a benchmark consisting of multiple samples.

    Attributes
    ----------
    samples: list, a list of Sample objects
    current_idx: int,  
    """
    def __init__(self, config: dict):
        self.samples = []
        self.current_idx = -1

        # Full config remains accessible for explicit calls
        self.config = config

        # Frequently accessed attributes assigned as class attributes for easier access
        self.benchmark_name = config['benchmark']['name']
        self.subset_size = config['benchmark']['subset_size']
        self.model_name = config['model']['name']
        self.run_name = config['experiments']['name']
        self.query_batch_size = config['batches']['query_batch_size']
        self.translation_batch_size = config['batches']['translation_batch_size']
        self.language_subset_name = config['experiments']['language_subset']
        
        # Languages based on the subset chosen in experiments
        self.languages = config["language_subsets"][self.language_subset_name]["iso_639_1"]

        # Optional: Metrics initialization
        self.metrics = {language: {} for language in self.languages}
        
        self.threadpooling = {
            "google_translator_delay": self.config['threadpool']['google_translator_delay'],
            "google_translator_max_workers": self.config['threadpool']['google_translator_max_workers'],
            "openai_delay": self.config['threadpool']['openai_delay'],
            "openai_max_workers": self.config['threadpool']['openai_max_workers']
        }

        # State tracking
        self.has_results = False
        self.has_translations = False

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

    def translate_all(self, data_type, save=True):
        for language in self.languages:
            logger.info(f"Starting async translation for {data_type} to {language}...")
        
            batches = self.create_batches(self.translation_batch_size)
            text_batches = prepare_translation_batches(batches, data_type)
            translator = self._get_translator(data_type, language)

            # Perform asynchronous translations in batches
            translated_batches=threadpool(
                data=text_batches,
                func=translate_batch,
                delay=self.threadpooling['google_translator_delay'],                # 2-second delay between requests
                max_workers=self.threadpooling['google_translator_max_workers'],    # Number of workers for threadpool
                translator=translator                                               # Specify if it's 'question' or 'output'
            )

            # Assign translated text to the corresponding sample attributes
            self._assign_translations(translated_batches, language, data_type)

            logger.info("Asynchronous translation completed successfully!")
        if save:
            self._save_translated_benchmark()

    def run(self, print_results=True, plot_results=False):
        """Main experiment function"""
        if not self.languages or not self.samples:
            raise ValueError("Please ensure this MultilingualBenchmark contains samples and target languages!")
        
        # Create from json
        new_instance = self.from_json(self.config)
        self.model_name = new_instance.model_name
        self.run_name = new_instance.run_name
        self.languages = new_instance.languages
        self.samples = new_instance.samples

        # 
        if not self.has_translations:            
            self.translate_all(data_type="question", save=True)

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
        self.write_to_json(f'{self.config["paths"]["results"]}/{self.run_name}.json')
    
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
 
    def _get_translator(self, data_type, target):
        """Helper function to initialize the appropriate translator."""
        if data_type == "question":
            return GoogleTranslator(target=target)
        elif data_type == "output":
            return GoogleTranslator(target='en')

    def _extract_texts(self, batch, data_type):
        """Helper function to extract the appropriate texts (questions or outputs) from the samples."""
        if data_type == "question":
            return [sample._to_question_str() for sample in batch]
        elif data_type == "output":
            return [sample._to_output_str() for sample in batch]

    def _assign_translations(self, translated_batches, target, data_type):
        """Helper function to assign translated texts to the corresponding sample attributes."""
        batch_size = self.translation_batch_size
        
        try:
            for i, batch in enumerate(translated_batches):
                for j, translated_text in enumerate(batch):
                    # Calculate the index in the samples list
                    index = i * batch_size + j

                    # Process the output if necessary
                    if data_type == "output":
                        translated_text = self.process_output_text(translated_text)
                    
                    # Assign the translation to the sample
                    try:
                        self.samples[index].assign(
                            value=translated_text,
                            language=target,
                            data_type=data_type
                        )
                    except Exception as e:
                        logger.warning(f"Error assigning translation: {e}. Assigning None to index {index}.")
                        self.samples[index].assign(
                            value=None,
                            language=target,
                            data_type=data_type
                        )
        except Exception as e:
            logger.error(f"Translation assignment failed for batch: {e}")



    def get_average_evaluation(self, language):
        """
        Calculates the average evaluation for a given language across all samples.

        Parameters
        ----------
        language: str
            The language for which the average evaluation is to be calculated.

        Returns
        -------
        float
            The average evaluation for the given language.
        """
        total_evaluation = 0
        count = 0
        for sample in self.samples:
            if language in sample.evaluations:
                total_evaluation += sample.evaluations[language]
                count += 1
        return total_evaluation / count if count > 0 else 0

    def get_all_evaluations(self):
        """
        Gets all evaluations for all languages and samples.

        Returns
        -------
        dict
            A dictionary where keys are languages and values are lists of evaluations for each sample.
        """
        evaluations_dict = {}
        for sample in self.samples:
            for language, evaluation in sample.evaluations.items():
                if language not in evaluations_dict:
                    evaluations_dict[language] = []
                evaluations_dict[language].append(evaluation)
        return evaluations_dict
    
    def create_batches(self, size):
        batches = []
        start_idx = 0

        for _ in range(len(self) // size):
            end_idx = start_idx + size
            batch = self.samples[start_idx:end_idx]
            batches.append(batch)
            start_idx = end_idx
        
        # Handle leftover samples if any
        if len(self) % size != 0:
            leftover_batch = self.samples[start_idx:]
            batches.append(leftover_batch)
        
        return batches
    
    def _assign_all_output(self, language, output: dict):
        """
        Takes dict: {0: Mars, 1: five, 2: etc.}, where 0 represents sample with idx 0 and Mars represents the output
        """
        # Sort benchmark such that we can
        for sample in self.samples: # TODO Substitute this with a non-looping solution
            if sample.idx in output:
                answer = output[sample.idx]
                sample.output[language] = answer

    # UNUSED?
    # def _assign_all_evaluations(self, target: str, evals: dict):
    #     """
    #     Assign evaluations to samples.
        
    #     Parameters:
    #         target (str): The evaluation target.
    #         evals (dict): Dictionary where keys are sample indices and values are evaluation results.
    #                 Example: {0: "mars", 1: "five", 2: "etc."}
    #     """
    #     for sample in self.samples:
    #         contains_nan = sample._contains_nan(target)
    #         sample.evaluations[target] = None if contains_nan else evals.get(sample.idx, None)
    
    def query_gpt(self):
        """
        Queries benchmark using all languages using GPT models
        """
        client = OpenAI()
        for language in self.languages:
            logger.info(f'Now querying {self.model_name} for language {language}')
            try:
                output = get_api_output(
                    self, 
                    language, 
                    api_type='openai', 
                    batch_size=self.config['batches']['query_batch_size'], 
                    client=client, 
                    engine=self.model_name, 
                    sys_prompt=self.config['prompts']['sys'])
                self._assign_all_output(language, output)
            except Exception as e:
                logger.critical(f"Error encountered querying {self.model_name} in language code {language}. Skipping language... \n ERROR: {e}")

    def query_gemini(self, engine='gemini-1.5-flash'):
        """
        Queries benchmark using all languages using Gemini models
        """
        for language in self.languages:
            # Translate system prompt
            sys_prompt = GoogleTranslator(target=language).translate(self.config['prompts']['sys'])
            
            model = genai.GenerativeModel(
                model_name=engine,
                system_instruction=sys_prompt,
                safety_settings=self.config['google_safety_levels']
            )

            logger.info(f'Now querying {self.model_name} for language {language}')
            try:
                output = get_api_output(self, language, api_type='google', batch_size=self.gpt_batch_size, model=model)
                self._assign_all_output(language, output)
            except Exception as e:
                logger.critical(f"Error encountered querying {self.model_name} in language code {language}. Skipping language... \n ERROR: {e}")

    def query_bloomz(self):
        """
        Queries benchmark using all languages using BLOOMZ models
        """
        for language in self.languages:
            logger.info(f'Now querying {self.model_name} for language {language}')
            try:
                output = get_local_model_output(
                    self, 
                    language, 
                    query_batch_size=self.config["experiments"]["query_batch_size"]
                )
                self._assign_all_output(language, output)
            except Exception as e:
                logger.critical(f"Error encountered querying {self.model_name} in language code {language}. Skipping language... \n ERROR: {e}")

    def evaluate_all(self):
        """Evaluate for all languages their answers using direct answer matching"""
        logger.info("Evaluating")
        for sample in self.samples:
            sample.evaluate()

    def _get_valid_evals(self, language):
        evals = [sample.evaluations.get(language, None) for sample in self.samples]
        return [float(eval) for eval in evals if (eval != 'null') and (eval is not None)]
    
    def _valid_evals_to_accuracy(self, valid_evals):
        # Calculate accuracy by excluding -1 responses
        valid_evaluations = [evaluation for evaluation in valid_evals if evaluation != -1]
        return sum(valid_evaluations) / len(valid_evaluations) if valid_evaluations else 0

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
            evaluations (dict): A dictionary with per-language accuracy
        """
        for language in self.languages:
            valid = self._get_valid_evals(language)
            acc = self._valid_evals_to_accuracy(valid)

            self.metrics[language].update({'a': acc})

        if save:
            self._save_metrics()

    def _save_metrics(self):
        # Define the path to the CSV file
        path = f"{self.config['paths']['metrics']}/metrics.csv"
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

    def _save_translated_benchmark(self):
        """Save translated QA pairs as .json - saves a lot of time during the experiment"""
        save_path = f'{self.config["paths"]["translated_benchmarks"]}/{self.benchmark_name}_{len(self.samples)}.json' 
        logger.info(f"Now saving translated benchmark to {save_path}")
        self.write_to_json(save_path)
        logger.info("Saved succesfully!")
            
    def _plot_results(self):
        if self.has_results:
            # Read CSV with metrics
            metrics_path = f"{self.config['paths']['metrics']}/metrics.csv"
            metrics = pd.read_csv(metrics_path).dropna()

            # Plot results
            plot(metrics)
            plot_de(metrics)
        else:
            logger.info('No results recorded yet for plotting. Please run cls.run() first.')


    @classmethod
    def from_json(cls, config):
        """Initialize class from configuration. Tries to load as much data as is already available based on configuration file. For example,
        if a translated benchmark exists, it tries to load this already. Likewise, if results exist, it tries to load"""
        # Initialize Benchmark object from JSON
        benchmark = cls(config)
        
        # Isolate model and languages
        model = benchmark.model_name
        languages = benchmark.languages
        
        # If run exists, load run. If run doesn't exist, load translated benchmark, if that doens't exist, just load non-translated benchmark
        results_path = f"{config['paths']['results']}/{config['experiments']['name']}.json"
        translated_path = f"{config['paths']['translated_benchmarks']}/{benchmark.benchmark_name}_{config['benchmark'][ 'subset_size']}.json"
        default_path = f"{config['paths']['benchmarks']}/{benchmark.benchmark_name}.json"
        
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                benchmark.has_results = True
        elif os.path.exists(translated_path):
            with open(translated_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                benchmark.has_translations = True
        else:
            with open(default_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        
        # Assign file contents to this benchmark's samples
        for i, item in enumerate(data):
            sample = Sample.from_dict(item, idx=i, languages=languages)
            benchmark.add_sample(sample)

        # Set results toggle to True
        benchmark.has_results = True

        return benchmark