## Getting started 
1. Download python3 by visiting [python official website](https://www.python.org/downloads/). 
2. Install the Biopython package. More information can be found in [biopython official website](https://biopython.org/wiki/Packages).
3. Download FASTQ files used in this experiments which can be accessed from Seqeuence Read Archive (SRA) with Bioproject: PRJNA670562.

## Computational pipeline
1. Split downloaded FASTQ files into multiple smaller files by running **SequenceSplit.py**. This software will automatically produce several FASTQ files with 20,000,000 reads in each file which makes parallelization much easier. 
2. Check read quality, collapse reads, and put them in a contingency-table-like data structure using **SequenceDecomplexationOptimized.py**. 
        **Recommendation** Please install parallel function to help with this multithreading. For mac users, you can use [Homebrew](https://brew.sh/) `brew install parallel`. Then run `parallel SequenceDecomplexationOptimized.py ::: group*.fastq`. 
3. Collapse multiple contingency tables into one file by running **SequenceDecomplexationTable.py**. 
4. **(Only for the first experiment)** Please run **SequenceDecomplexationTableExtra.py** to improve the qualities of some reads. 
5. The resulting pickle and csv files are ready for the subsequent downstream analyses. 

**Note** Please add fastq filename and output filename in these files before use to make them work properly.    

