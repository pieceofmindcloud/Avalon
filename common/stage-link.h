// A model split by layers across processes: the link between two pipeline stages.
//
// Each process loads its own part of the model (a slice of its layers). The last stage serves the requests; every
// earlier stage runs as a stage server (llama-stage-server) that hands the next stage the hidden state its last layer
// produces for every token. The later stage connects to the earlier one and, for each decode, sends the tokens and
// takes their rows back (llama_set_input_rows_cb); every operation on its memory is mirrored there too
// (llama_set_memory_op_cb), so both KV caches hold the same sequences.
//
// Frames: a little-endian u32 length, then the payload. A payload starts with a kind byte: 1, bytes as they are; 2, a
// tensor: u32 n_meta, n_meta bytes of meta, then f32 values. Requests are bytes and start with an op; answers start
// with a status byte (0 ok, 1 error then the message), and the rows of 'P' or 'W' go as a tensor whose meta is the
// status byte, so a relay between the stages may carry them in another precision and hand them back as f32.
//
//   'H' token                                       -> u32 width             (the first request of a connection)
//   'P' i64 pos0, u32 n, i32 tok[n], i32 pos[n], i32 seq[n]   -> f32 rows[n * width]   (a prompt piece, n > 1)
//   'W' i64 pos0, u32 n, i32 tok[n], i32 pos[n], i32 seq[n]   -> f32 rows[n * width]   (generation, a few tokens)
//   'M' u8 op, i32 seq, i32 seq_dst, i32 p0, i32 p1, i32 shift -> u8 result     (llama_memory_op)
//   'Q'                                             -> (the stage server ends)
//
// The stage server answers each connection's requests in order.

#pragma once

#include "llama.h"
#include "../src/llama-ext.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace stage_link {

constexpr uint8_t KIND_BYTES  = 1;
constexpr uint8_t KIND_TENSOR = 2;

// "host:port" -> a connected (listen = false) or listening socket; -1 with err
int tcp_open(const std::string & addr, bool listen, std::string & err, int * bound_port = nullptr);

bool write_frame(int fd, const std::vector<uint8_t> & payload);
// false at the end of the stream or on an error
bool read_frame(int fd, std::vector<uint8_t> & payload);

// constant-time comparison of two secrets
bool same_secret(const std::string & a, const std::string & b);

// the later stage's side of one connection
class client {
public:
    ~client();

    // connects and presents the token; the width of the rows the earlier stage hands on
    bool connect(const std::string & addr, const std::string & token, std::string & err);
    int32_t width() const { return width_; }

    // the earlier stage's rows for n tokens (n * width floats)
    bool rows(int32_t n, const llama_token * tok, const llama_pos * pos, const llama_seq_id * seq, float * out,
              int32_t width, std::string & err);
    bool memory(llama_memory_op op, llama_seq_id seq, llama_seq_id seq_dst, llama_pos p0, llama_pos p1,
                llama_pos shift, std::string & err);

    // a context of this process asks this client for its input rows and mirrors its memory through it
    void attach(llama_context * ctx);

private:
    bool call(const std::vector<uint8_t> & request, std::vector<uint8_t> & answer, std::string & err);

    int fd_ = -1;
    int32_t width_ = 0;
    std::mutex mu_;
};

// the earlier stage's side: serves the connections on addr with ctx until 'Q' or a fatal error
int serve(llama_context * ctx, const std::string & addr, const std::string & token, int32_t n_batch);

} // namespace stage_link
