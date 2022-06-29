## Srom Installation Instruction

### Install srom in Jupyter Notebook.

1. Create an access token for artifactory:

- first
- second
- third

2. Create a cell and run the follow command line in the Jupyter Notebook:
    ```
    !pip install --extra-index-url https://<<Your-IBM-Email-ID>>:<<Artifactory-Token>>@na.artifactory.swg-devops.com/artifactory/api/pypi/res-srom-pypi-local/simple srom[deep_learning]==1.5.10
    ```
    where **\<<Your-IBM-Email-ID\>>** is your IBM Email ID and **\<<Artifactory-Token\>>** is your API key that you can access follow the step 1.
 
