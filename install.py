import subprocess

# Define the list of libraries to install
libraries = ['networkx','pandas', 'tqdm', 'python-dotenv', 'scikit-learn', 'matplotlib']

# Use subprocess to run pip install command for each library
for library in libraries:
    try:
        subprocess.check_call(['pip', 'install', library])
        print(f'Successfully installed {library}.')
    except subprocess.CalledProcessError as e:
        print(f'Error installing {library}: {e}')