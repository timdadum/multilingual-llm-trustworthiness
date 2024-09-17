import logging
import sys
import codecs

# Set up UTF-8 encoding for the stream (console) handler
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Create a custom logger
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('logger.log', encoding='utf-8')  # Ensure UTF-8 encoding for the file

# Set minimum logging level
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

# Format logging output
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set formatter
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handler to logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)