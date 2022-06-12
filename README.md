# PINN-Voltammetry-Best-Practices
This is a code repository for 'A Critical Evaluation of Using Physics-Informed Neural Networks for Simulating Voltammetry: Strengths, Weaknesses and Best Practices' submitted to *Journal of Electroanalytical Chemistry*.

# Requirements
Python 3.7 and above is suggested to run the program. The neural networks was developed and tested with Tensorflow 2.3. To install required packages, run

```
$ pip install -r requirement.txt

```
# Eight Test Cases
The code repository contains eight folders, for the eight test cases mentioned in paper. 

* Chronoamperometry at a Macro Electrode: Highlights the importance of non-zero conditioning time.
* Chronoamperometry at a Spherical Electrode: Mathematical transformation of PDEs to an easier form may increase the performance of PINN. 
* Cyclic voltammetry at a Spherical Electrode: Sequence to sequence training for simulation of long time duration. 
* Cyclic voltammetry at a Macro Electrode: Adaptive weights algorithm.
* Chronoamperometry at a microband electrode: effect of batch size on learning.
* Chronoamperometry at a cube electrode: overlapping domain decomposition increases the accuracy of prediction. 
* Cyclic voltammetry at a cube electrode: Effects of learning rate scheduling on accuracy of prediction. Please note that running this program needs a very large RAM (~ 90 GB)
* Chronoamperometry at a microdisc electrode: Switch from cylindrical coordinates to Cartesian coordinates when facing gradient problems. 



# Issue Reports
Please report any issues/bugs of the code in the discussion forum of the repository or contact the corresponding author of the paper


# Cite
To cite, please refer to: (Cite unavailable yet)

