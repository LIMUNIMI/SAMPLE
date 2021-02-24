# Spectral Analysis for Modal Parameter Linear Estimate
## Abstract
Modal synthesis is used to generate the sounds associated with the vibration of rigid bodies, according to the characteristics of the force applied onto the object. Towards obtaining sounds of high quality, a great quantity of modes is necessary, the development of which is a long and tedious task for sound designers as they have to manually write the modal parameters.
This paper presents a new approach for practical modal parameter estimation based on the spectral analysis of a single audio example. The method is based on modelling the spectrum of the sound with a time-varying sinusoidal model and fitting the modal parameters with linear and semi-linear techniques.
We also detail the physical and mathematical principles that motivate the algorithm design choices.
A Python implementation of the proposed approach has been developed and tested on a dataset of impact sounds considering objects of different shapes and materials. We assess the performance of the algorithm by evaluating the quality of the resynthesised sounds. Resynthesis is carried out via the Sound Design Toolkit (SDT) modal engine and compared to the sounds resynthesised from parameters extracted by SDT's own estimator. The proposed method was thoroughly evaluated both objectively using perceptually relevant features and subjectively following the MUSHRA protocol.

## PDF
Your can find the paper in the SMC 2020 proceedings [here](https://smc2020torino.it/adminupload/file/SMCCIM_2020_paper_167.pdf).

## Cite
```
@inproceedings{tiraboschi2020spectral,
	title={Spectral Analysis for Modal Parameters Linear Estimate},
	author={Tiraboschi, M and Avanzini, F and Ntalampiras, S},
	booktitle={Sound \& Music Computing Conference},
	year={2020},
	organization={SMC},
}
```