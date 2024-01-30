# Image Processing Lab Installation Instructions

## Requirements
This project is developed using Python 3.8. Ensure you have Python 3.8 installed on your system.

## Installation Steps

1. Clone this repository or download the project files to your local machine.Run the following command:
    ```shell
    git clone https://github.com/sugnite/ISAE-Image-Processing-Lab-HAD501.git
    ```
2. Open a terminal or command prompt and navigate to the directory where the project files are located.
    
    ```shell
    cd ISAE-Image-Processing-Lab-HAD501
    ```

3. Install the required packages using pip. Run the following command:
    ```shell
    pip3.8 install -r requirements.txt --user
    ```

    This will install the specific versions of numpy (1.17), matplotlib (3.5), and OpenCV required for the project in your Python 3.8 environment. 
    The `--user` flag installs the packages locally for the current user, avoiding the need for administrative privileges.

4. Once the installation is complete, you can run the `jupyter notebook` with all the lab's instructions
    ```
    ~/.local/bin/jupyter-notebook img_proc_lab.ipynb
    ```

5. For the documentation of the OpenCV Binding, you can open this HTML doc about `LabImageProcessing` which is an OpenCV Binding with all the function you need for exercise 1 to 3
    ```shell
    firefox docs/_build/html/OpenCvBinding.html
    ```

