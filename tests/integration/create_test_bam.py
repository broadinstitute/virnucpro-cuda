#!/usr/bin/env python3
"""Create test BAM file for integration tests."""
import pysam

bam_path = 'test_data/small.bam'
header = {'HD': {'VN': '1.0', 'SO': 'unsorted'}}

with pysam.AlignmentFile(bam_path, 'wb', header=header) as bam:
    sequences = [
        'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG',
        'GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA',
        'TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA',
        'CGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGC',
        'AGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTC',
        'TGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGAC',
        'CAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGT',
        'GTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCA',
        'TCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAG',
        'ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG',
        'CTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGA',
        'GACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTG',
        'ACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGA',
        'CTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGAC',
        'TTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACTTGACT',
    ]

    for i, seq in enumerate(sequences):
        read = pysam.AlignedSegment()
        read.query_name = f'read{i}'
        read.query_sequence = seq
        read.flag = 4
        read.reference_id = -1
        read.reference_start = -1
        read.mapping_quality = 0
        read.cigar = None
        read.next_reference_id = -1
        read.next_reference_start = -1
        read.template_length = 0
        read.query_qualities = pysam.qualitystring_to_array('I' * len(seq))
        bam.write(read)

empty_bam_path = 'test_data/empty.bam'
with pysam.AlignmentFile(empty_bam_path, 'wb', header=header) as bam:
    pass

print(f'Created {bam_path} with 15 reads')
print(f'Created {empty_bam_path} (empty)')
