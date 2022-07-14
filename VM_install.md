## Virtual Machine Setup Instruction

### Require VM from IaaS.
1. Go to website: https://draco.bx.cloud9.ibm.com/iaas/Devices
2. Click "Create New Virtual Machine" button on the left to create a new VM
3. Fill in the VM configuration information:
- **Hostname:** Any name you like, could be "machine01", etc.
- **Datacenter:** Choose "dal13 - Dallas 13"
- **Flavors:** Choose ("Name", "VCPU", "RAM", "First Disk") = ("B1_32X64X25", "32 x 2.0 GHz or higher Cores", "64 GB", "25 GB (SAN)")
- **Export Classification:** Choose "Blue"
- **Data Classification:** Choose "Anonymised Data"
5. Click "Submit" buttton, would be redirected to the main page automatically
6. Wait about 5 minutues for the VM "IBM IP Address" being assigned
7. Click the newly created VM's "Device Name", go to the "Instance Detail" page of this VM

### Connect to VM.
1. Open the terminal, use "ssh" to connect to the VM. The **\<IBM IP\>** is the IP address on the "Instance Detail" page of correspond VM, of the format 'X.XX.XXX.XXX' where 'X' are number.
```
ssh root@<IBM IP>
```
2. Terminal would prompt following message. Type "yes".
```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```
3. Then prompt following message. Type in the **"Unix Root Password"** on the "Instance Detail" page.
```
root@X.XX.XXX.XXX's password:
```
4. First time connection would require change the password. Type **"Unix Root Password"** again, then type and retype your new password.
```
You are required to change your password immediately (root enforced)
Changing password for root.
(current) UNIX password: 
New password: 
Retype new password:
```
5. Now, you have been successfully connected to the VM.

### Setup VM evironment.

1. Python Setup
```
yum install nano -y
sudo yum -y update
sudo yum -y install ssh
sudo yum -y groupinstall "Development Tools"
sudo yum -y install openssl-devel bzip2-devel libffi-devel
sudo yum install gcc-c++
sudo yum install -y xz-devel
sudo yum install lzma -y
yum install screen -y
sudo yum -y install wget
wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz
tar xvf Python-3.9.7.tgz
cd Python-3.9*/
./configure --enable-optimizations
sudo make altinstall
cd ..
pip3.9 install --upgrade pip
sudo yum install java-1.8.0-openjdk-devel -y
```

2. Spark Installation
```
mkdir ~/srom_spark
cd ~/srom_spark
wget https://archive.apache.org/dist/spark/spark-3.0.2/spark-3.0.2-bin-hadoop3.2.tgz
tar -xzf spark-3.0.2-bin-hadoop3.2.tgz
echo 'export SPARK_HOME=~/srom_spark/spark-3.0.2-bin-hadoop3.2' >> ~/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
cd ~/
pip install pyspark==3.0.2
echo 'export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip' >> ~/.bashrc
source ~/.bashrc
echo 'export PYSPARK_PYTHON=python3.9' >> ~/.bashrc
source ~/.bashrc
```

3. SROM Installation
3.1. Create/Access an API Key for artifactory:

- Go to JFrog Platform with the link https://na.artifactory.swg-devops.com/ . Normally, you would be automatically login with your IBM ID by accessing this link.
- On the right corner of the page is your IBM Email ID:<br />
![image01](https://github.com/syhAnna/Model-Inspection/blob/main/images/01.jpg)
- Then, click this "Welcome, ...", go to the "Edit Profile" to access the user profile:<br />
![image02](https://github.com/syhAnna/Model-Inspection/blob/main/images/02.jpg)
- Finally, you can see "API Key" in "Authentication Settings" part, copy this API Key to clipboard then you can use your API Key now:<br />
![image03](https://github.com/syhAnna/Model-Inspection/blob/main/images/03.jpg)


3.2. Install srom by running the following command
    ```
    pip install --extra-index-url https://<<Your-IBM-Email-ID>>:<<Artifactory-Token>>@na.artifactory.swg-devops.com/artifactory/api/pypi/res-srom-pypi-local/simple srom[deep_learning]==1.5.10
    ```
    where **\<<Your-IBM-Email-ID\>>** is your IBM Email ID and **\<<Artifactory-Token\>>** is your API key that you can access follow the step 3.1.
    
### Install and use screen.
With the Linux screen command, you can push running terminal applications to the background and pull them forward when you want to see them. It also supports split-screen displays and works over SSH connections, even after you disconnect and reconnect.

1. Install **screen** use this command:
```
yum install screen -y
```

2. Use **screen**:
To start screen, simply type it as shown below and hit Enter:
```
screen
```
- Type Ctrl+A, release those keys, and then press d to detach the screen. The download process is still running but the window showing the download is removed. 
- You need the number from the start of the window name to reattach it, to get a list of the detached windows:
```
screen -ls
```
- Then, you can use the -r (reattach) option and the number of the session to reattach it, for example:
```
screen -r 23167
```
- When the process ends, you can type exit to exit from the screen. Alternatively, you can press Ctrl+A, and then K to forcibly kill a window:
```
exit
```
- More screen related commands refer to: https://www.howtogeek.com/662422/how-to-use-linuxs-screen-command/






