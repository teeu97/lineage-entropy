"""
SequenceSplit.py splits a FASTQ file into multiple ones that contain 20,000,000 reads each. This split helps with
the downstream parallelization of decomplexation process
"""

from Bio import SeqIO

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = iterator.__next__()
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch

fastq_filename = ''  # add fastq filename here

record_iter = SeqIO.parse(fastq_filename,"fastq")
for i, batch in enumerate(batch_iterator(record_iter, 20000000)):
    filename = "group_%i.fastq" % (i + 1)  # the output filename is group_i.fastq 
    with open(filename, "w") as handle:
        count = SeqIO.write(batch, handle, "fastq")
    print("Wrote %i records to %s" % (count, filename))
