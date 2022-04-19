#include <albp/read_fasta.hpp>
#include <albp/antidiag.hpp>

#include <limits>
#include <iostream>

#include "omp.h"

int main(int argc, char **argv)
{
    std::string q = "query_batch.fasta";
    std::string t = "target_batch.fasta";

    if (argc == 3)
    {
        q = argv[1];
        t = argv[2];
    } 

    // std::cout << "filename minlen maxlen avglen total_bytes\n";
    albp::FastaInput query = albp::ReadFasta(q.c_str());
    albp::FastaInput target = albp::ReadFasta(t.c_str());

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
    // printf("query sequence count = %ld\n", query.sequence_count());
    // printf("targe sequence count = %ld\n", target.sequence_count());


    //     int x = 0, y = 0;
    //     int res = albp::antidiag("AAA", "TAA", 6, 1, 1, -4, &x, &y);
    //     printf("%d\t%d\t(%d, %d)(%ld-%ld)\n", 0, res, x, y, 3, 3);

    // return 0;

    std::vector<std::pair<int, int>> scores;

    uint inf = 0;
    uint sup = 10;
    
    for (uint i = inf; i < sup; i++)
    {
        int x = 0, y = 0;
        int res = albp::antidiag(query.sequences.at(i), target.sequences.at(i), 6, 1, 1, -4, &x, &y);
        printf("%d\t%d\t(%d, %d)(%ld-%ld)\n", i, res, x, y, query.sequences.at(i).length(), target.sequences.at(i).length());
    }

    return 0;
}