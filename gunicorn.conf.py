# gunicorn.conf.py

# Do not preload the application so that each worker instantiates its own MongoClient.
preload_app = False

# Set number of workers based on the number of CPU cores.
workers = 2  # or more, depending on available resources

# Increase timeout to allow slower responses.
timeout = 120

# Bind address can be set if needed:
bind = "0.0.0.0:5000"

# Log level
loglevel = "debug"
