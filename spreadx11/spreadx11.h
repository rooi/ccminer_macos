#define MAX_BLOCK_SIZE 200000

struct spreadwork {
        unsigned char data[185];
        unsigned char privkey[32];
        unsigned char kinv[32];
        unsigned char tx[MAX_BLOCK_SIZE];
        size_t txsize;
};
