#include <albp/read_fasta.hpp>
#include <albp/simple_sw.hpp>

#include <limits>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << argv[0] << " <Fastq File> <Fasta File> [Fasta File...]" << std::endl;
        return -1;
    }

    std::cout << "filename minlen maxlen avglen total_bytes\n";
    albp::FastaInput query = albp::ReadFasta(argv[1]);
    albp::FastaInput target = albp::ReadFasta(argv[2]);

    for (int i = 1; i < argc; i++)
    {
        const auto fasta = albp::ReadFasta(argv[i]);

        size_t minlen = std::numeric_limits<size_t>::max();
        size_t maxlen = 0;
        size_t total_len = 0;

        for (const auto &x : fasta.sequences)
        {
            minlen = std::min(minlen, x.size());
            maxlen = std::max(maxlen, x.size());
            total_len += x.size();
        }

        std::cout << argv[i]
                  << " " << fasta.sequence_count()
                  << " " << minlen
                  << " " << maxlen
                  << " " << (total_len / fasta.sequence_count())
                  << " " << total_len
                  << "\n";
    }
    printf("query sequence count = %d\n", query.sequence_count());
    printf("targe sequence count = %d\n", target.sequence_count());

    for (uint i = 0; i < query.sequence_count(); i++)
    {
        albp::SimpleSmithWatermanResult res = albp::simple_smith_waterman(query.sequences[i], target.sequences[i], -1, -2, 5, -2);
        printf("%d\t%d\n", i, res.score);        
    }

    return 0;
}