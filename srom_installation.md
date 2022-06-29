## Srom Installation Instruction

### Install srom in Jupyter Notebook.

1. Create/Access an API Key for artifactory:

- Go to JFrog Platform with the link https://na.artifactory.swg-devops.com/ . Normally, you would be automatically login with your IBM ID by access this link.
- On the right corner of the page is your IBM Email ID:<br />
![image01](https://github.com/syhAnna/Model-Inspection/blob/main/images/01.jpg)
- Then, click this "Welcome, ...", go to the "Edit Profile" to access the user profile:<br />
![image02](https://github.com/syhAnna/Model-Inspection/blob/main/images/02.jpg)
- Finally, you can see the "API Key" in the "Authentication Settings" part, copy this API Key to clipboard then you can use your API Key now:<br />
![image03](https://github.com/syhAnna/Model-Inspection/blob/main/images/03.jpg)


2. Create a cell and run the follow command line in the Jupyter Notebook:
    ```
    !pip install --extra-index-url https://<<Your-IBM-Email-ID>>:<<Artifactory-Token>>@na.artifactory.swg-devops.com/artifactory/api/pypi/res-srom-pypi-local/simple srom[deep_learning]==1.5.10
    ```
    where **\<<Your-IBM-Email-ID\>>** is your IBM Email ID and **\<<Artifactory-Token\>>** is your API key that you can access follow the step 1.
 
3. Check you have install srom by import it:
    ```
    import srom
    ```
