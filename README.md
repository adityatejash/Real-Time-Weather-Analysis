NOTE: In line 83 of "Project.py", copy the path of "weather.csv" file and replace it with the copied path. Also add "\" after every "\".

Step 1: Download the repository as a ZIP file and extract it to your desired location.

Step 2: Install Python (if not already installed) and ensure pip is available.

Step 3: Open PowerShell and run the following command to allow running scripts for the current user:
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Step 4: Navigate to the project folder in PowerShell and create a virtual environment:
        python -m venv myenv

Step 5: Activate the virtual environment:
        .\myenv\Scripts\Activate.ps1

Step 6: Install the required Python libraries:
        pip install requests pandas numpy scikit-learn pytz

Step 7: Run the project.py file

Step-8: Deactivate the virtual environment:
        deactivate
