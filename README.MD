# Flask Image Prediction API

## Description

This is a Flask-based image prediction API for a college project. It uses a pre-trained Vision Transformer (ViT) model to classify images.

## Prerequisites

Before cloning the project, make sure you have the following installed:

- [Git](https://git-scm.com/)
- [Git LFS (Large File Storage)](https://git-lfs.github.com/)

### Install Git LFS on Windows

1. **Download Git LFS**:
   - Go to the [Git LFS releases page](https://git-lfs.github.com/) and download the Windows installer.
   
2. **Run the installer**:
   - Once downloaded, run the `.exe` installer and follow the installation instructions.

3. **Initialize Git LFS**:
   - Open a Command Prompt (you can search for `cmd` in the Start menu).
   - Run the following command to initialize Git LFS:
     ```bash
     git lfs install
     ```

## Cloning the Repository

Follow these steps to clone the repository and get started with your project.

1. **Clone the repository**:
   - Open Command Prompt and navigate to the folder where you want to clone the repository.
   - Run the following command to clone the repo:
     ```bash
     git clone https://github.com/Primarizki21/FinalProjectFlaskApp.git
     ```

2. **Navigate to the project folder**:
   - After cloning, navigate to the project directory:
     ```bash
     cd your-repository
     ```

3. **Pull large files managed by Git LFS**:
   - Git LFS manages large files separately from the rest of your code. After cloning the repository, you need to fetch these large files by running:
     ```bash
     git lfs pull
     ```

   This command ensures that all the large files stored using Git LFS are downloaded and available for use in your project.

### Creating Virtual Environment

Our project need a Python Virtual Environment to run this. It is optional, but it would be highly recommended to ensure that the correct dependencies and Python version are used without interfering with other projects or system-wide installations.

1. **Navigate to the Project Folder**:
    - Navigate into project directory, like 1 level above the project:
    ```bash
    cd path/to/your/workingdir
    ```

2. **Create a Virtual Environment**:
    - It’s best to use a virtual environment to manage dependencies. Create a new virtual environment by running:
    ```bash
    python -m venv your-venv-name
    ```

3. **Activate the virtual environment**:
    - **On Windows**, run:
    ```bash
    your-venv-name\Scripts\activate
    ```

## Running the Project

Before running the project, make sure to install the dependencies.

1. **Install dependencies**:
    - Install all the required Python packages using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Project**:
    - You can run the project by running:
    ```bash
    python folder-project-name\run.py
    ```

3. **Stop the Project**:
    - You can press `Ctrl + C` to stop the process: