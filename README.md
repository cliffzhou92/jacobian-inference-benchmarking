# Benchmarking spliceJAC with existing GRN inference methods using Beeline

## Introduction:
This repository stores the scripts, input datasets and output files to reproduce GRN inference benchmarking results in our spliceJAC preprints.

We used [Beeline](https://murali-group.github.io/Beeline/BEELINE.html) from Murali lab to compare our spliceJAC method with 12 existing GRN inference methods to infer celltype-specific regulations on four synthetic datasets from 1) EMT circuits using our simulation, and 2) cycle 3) converging bifurcation 4)trifurcation GRN circuits generated by [BoolODE](https://murali-group.github.io/Beeline/BoolODE.html) package.

## Steps to reproduce the benchmarking results
1. Download and configure the Beeline package ([latest version from Murali lab](https://github.com/Murali-group/Beeline), or the [forked repo](https://github.com/cliffzhou92/Beeline) that we have been using). Build the docker files as instructed.
1. Move the `./inputs` and `./config_files` folder in this repo to the root directory of Beeline. The `./inputs` files contains gene expression and pseudo-time files for each cell type in synthetic datasets, which are the required inputs in Beeline.
1. Run , where FILENAME is one of the yaml files in current './config_files' folder to select which dataset to run benchmarking on. For documentation purpose, we have also uploaded all the generated results in `./output` directory of this repo.
1. (Optional) To include spliceJAC in Beeline performance metric statistics, move the subdirectory named spliceJAC in `./output` directory of this repo (which contains the rankedEdges.csv for spliceJAC inferred results) into the corresponding output directory of Beeline as in parallel to the subdirectory of other algorithm outputs. In addition, modify the yaml file to change the "should_run" parameter in spliceJAC from False to True.
1. Run in command lines to generate performance statistics, which will appear in the `./output` folder in Beeline root directory. For documentation, we have also uploaded all the generated results in `./output` directory of this repo.
