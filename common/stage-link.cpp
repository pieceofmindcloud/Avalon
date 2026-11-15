#include "stage-link.h"

#include "../src/llama-ext.h"
#include "log.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <thread>

#if defined(_WIN32)
// the stage link is not built on Windows yet
namespace stage_link {
int tcp_open(const std::string &, bool, std::string & err, int *) { err = "the stage link needs POSIX sockets"; return -1; }
bool write_frame(int, const std::vector<uint8_t> &) { return false; }
bool read_frame(int, std::vector<uint8_t> &) { return false; }
bool same_secret(const std::string & a, const std::string & b) { return a == b; }
client::~client() {}
bool client::connect(const std::string &, const std::string &, std::string & err) { err = "not supported"; return false; }
bool client::rows(int32_t, const llama_token *, const llama_pos *, const llama_seq_id *, float *, int32_t, std::string & err) { err = "not supported"; return false; }
bool client::memory(llama_memory_op, llama_seq_id, llama_seq_id, llama_pos, llama_pos, llama_pos, std::string & err) { err = "not supported"; return false; }
void client::attach(llama_context *) {}
bool client::call(const std::vector<uint8_t> &, std::vector<uint8_t> &, std::string & err) { err = "not supported"; return false; }
int serve(llama_context *, const std::string &, const std::string &, int32_t) { return 1; }
}
#else

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace stage_link {

// a frame is a prompt piece's rows at most
static constexpr uint32_t MAX_FRAME = 1u << 30;

template <typename T> static void put(std::vector<uint8_t> & out, T v) {
    const auto * p = reinterpret_cast<const uint8_t *>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T> static bool get(const std::vector<uint8_t> & in, size_t & at, T & v) {
    if (at + sizeof(T) > in.size()) {
        return false;
    }
    std::memcpy(&v, in.data() + at, sizeof(T));
    at += sizeof(T);
    return true;
}

int tcp_open(const std::string & addr, bool listen, std::string & err, int * bound_port) {
    const size_t colon = addr.rfind(':');
    if (colon == std::string::npos) {
        err = "expected HOST:PORT, not " + addr;
        return -1;
    }
    std::string host = addr.substr(0, colon);
    const std::string port = addr.substr(colon + 1);
    if (!host.empty() && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (listen) {
        hints.ai_flags = AI_PASSIVE;
    }
    addrinfo * res = nullptr;
    if (const int rc = getaddrinfo(host.empty() ? nullptr : host.c_str(), port.c_str(), &hints, &res); rc != 0) {
        err = "cannot resolve " + addr + ": " + gai_strerror(rc);
        return -1;
    }
    int fd = -1;
    for (addrinfo * a = res; a && fd < 0; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) {
            continue;
        }
        const int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        if (listen) {
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
            if (bind(fd, a->ai_addr, a->ai_addrlen) != 0 || ::listen(fd, 16) != 0) {
                close(fd);
                fd = -1;
                continue;
            }
            if (bound_port) {
                sockaddr_storage ss{};
                socklen_t sl = sizeof ss;
                getsockname(fd, (sockaddr *) &ss, &sl);
                *bound_port = ss.ss_family == AF_INET6 ? ntohs(((sockaddr_in6 *) &ss)->sin6_port)
                                                       : ntohs(((sockaddr_in *) &ss)->sin_port);
            }
        } else if (::connect(fd, a->ai_addr, a->ai_addrlen) != 0) {
            close(fd);
            fd = -1;
        }
    }
    freeaddrinfo(res);
    if (fd < 0) {
        err = std::string(listen ? "cannot listen on " : "cannot connect to ") + addr + ": " + std::strerror(errno);
    }
    return fd;
}

static bool write_all(int fd, const uint8_t * p, size_t n) {
    while (n > 0) {
        const ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) {
            if (w < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        p += w;
        n -= (size_t) w;
    }
    return true;
}

static bool read_all(int fd, uint8_t * p, size_t n) {
    while (n > 0) {
        const ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        p += r;
        n -= (size_t) r;
    }
    return true;
}

bool write_frame(int fd, const std::vector<uint8_t> & payload) {
    const uint32_t len = (uint32_t) payload.size();
    uint8_t head[4];
    std::memcpy(head, &len, 4);
    return write_all(fd, head, 4) && write_all(fd, payload.data(), payload.size());
}

bool read_frame(int fd, std::vector<uint8_t> & payload) {
    uint8_t head[4];
    if (!read_all(fd, head, 4)) {
        return false;
    }
    uint32_t len = 0;
    std::memcpy(&len, head, 4);
    if (len > MAX_FRAME) {
        return false;
    }
    payload.resize(len);
    return read_all(fd, payload.data(), len);
}

bool same_secret(const std::string & a, const std::string & b) {
    if (a.size() != b.size()) {
        return false;
    }
    uint8_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        diff |= (uint8_t) (a[i] ^ b[i]);
    }
    return diff == 0;
}

// an answer's status and its data: a bytes frame (status, data...) or a tensor frame (meta = status, then f32)
static bool parse_answer(const std::vector<uint8_t> & frame, std::vector<uint8_t> & data, std::vector<float> * values,
                         std::string & err) {
    if (frame.empty()) {
        err = "an empty answer";
        return false;
    }
    if (frame[0] == KIND_TENSOR) {
        size_t at = 1;
        uint32_t n_meta = 0;
        if (!get(frame, at, n_meta) || at + n_meta > frame.size() || n_meta < 1) {
            err = "a malformed tensor answer";
            return false;
        }
        const uint8_t status = frame[at];
        at += n_meta;
        if (status != 0) {
            err = "the earlier stage failed";
            return false;
        }
        if (values) {
            const size_t n = (frame.size() - at) / sizeof(float);
            values->resize(n);
            std::memcpy(values->data(), frame.data() + at, n * sizeof(float));
        }
        data.clear();
        return true;
    }
    if (frame[0] != KIND_BYTES || frame.size() < 2) {
        err = "a malformed answer";
        return false;
    }
    if (frame[1] != 0) {
        err = "the earlier stage: " + std::string(frame.begin() + 2, frame.end());
        return false;
    }
    data.assign(frame.begin() + 2, frame.end());
    return true;
}

client::~client() {
    if (fd_ >= 0) {
        close(fd_);
    }
}

bool client::call(const std::vector<uint8_t> & request, std::vector<uint8_t> & answer, std::string & err) {
    if (fd_ < 0) {
        err = "not connected to the earlier stage";
        return false;
    }
    if (!write_frame(fd_, request) || !read_frame(fd_, answer)) {
        err = "the link to the earlier stage broke";
        close(fd_);
        fd_ = -1;
        return false;
    }
    return true;
}

bool client::connect(const std::string & addr, const std::string & token, std::string & err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ >= 0) {
        close(fd_);
    }
    fd_ = tcp_open(addr, false, err);
    if (fd_ < 0) {
        return false;
    }
    std::vector<uint8_t> req{KIND_BYTES, 'H'};
    req.insert(req.end(), token.begin(), token.end());
    std::vector<uint8_t> frame, data;
    if (!call(req, frame, err) || !parse_answer(frame, data, nullptr, err)) {
        return false;
    }
    size_t at = 0;
    uint32_t width = 0;
    if (!get(data, at, width) || width == 0) {
        err = "the earlier stage's hello is malformed";
        return false;
    }
    width_ = (int32_t) width;
    return true;
}

bool client::rows(int32_t n, const llama_token * tok, const llama_pos * pos, const llama_seq_id * seq, float * out,
                  int32_t width, std::string & err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (width != width_) {
        err = "the earlier stage hands on rows of " + std::to_string(width_) + " floats, this one reads " +
              std::to_string(width);
        return false;
    }
    std::vector<uint8_t> req{KIND_BYTES, (uint8_t) (n > 1 ? 'P' : 'W')};
    put<int64_t>(req, n > 0 ? (int64_t) pos[0] : 0);
    put<uint32_t>(req, (uint32_t) n);
    for (int32_t i = 0; i < n; ++i) put<int32_t>(req, tok[i]);
    for (int32_t i = 0; i < n; ++i) put<int32_t>(req, pos[i]);
    for (int32_t i = 0; i < n; ++i) put<int32_t>(req, seq[i]);
    std::vector<uint8_t> frame, data;
    std::vector<float> values;
    if (!call(req, frame, err) || !parse_answer(frame, data, &values, err)) {
        return false;
    }
    if (values.size() != (size_t) n * width) {
        err = "the earlier stage gave " + std::to_string(values.size()) + " floats for " + std::to_string(n) + " rows";
        return false;
    }
    std::memcpy(out, values.data(), values.size() * sizeof(float));
    return true;
}

bool client::memory(llama_memory_op op, llama_seq_id seq, llama_seq_id seq_dst, llama_pos p0, llama_pos p1,
                    llama_pos shift, std::string & err) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<uint8_t> req{KIND_BYTES, 'M', (uint8_t) op};
    put<int32_t>(req, seq);
    put<int32_t>(req, seq_dst);
    put<int32_t>(req, p0);
    put<int32_t>(req, p1);
    put<int32_t>(req, shift);
    std::vector<uint8_t> frame, data;
    if (!call(req, frame, err) || !parse_answer(frame, data, nullptr, err)) {
        return false;
    }
    return !data.empty() && data[0] == 1;
}

static bool client_rows_cb(void * user, int32_t n, const llama_token * tok, const llama_pos * pos,
                           const llama_seq_id * seq, float * rows, int32_t width) {
    std::string err;
    if (!static_cast<client *>(user)->rows(n, tok, pos, seq, rows, width, err)) {
        LOG_ERR("stage link: %s\n", err.c_str());
        return false;
    }
    return true;
}

static bool client_memory_cb(void * user, llama_memory_op op, llama_seq_id seq, llama_seq_id seq_dst, llama_pos p0,
                             llama_pos p1, llama_pos shift) {
    std::string err;
    const bool ok = static_cast<client *>(user)->memory(op, seq, seq_dst, p0, p1, shift, err);
    if (!ok && !err.empty()) {
        LOG_ERR("stage link: %s\n", err.c_str());
    }
    return ok;
}

void client::attach(llama_context * ctx) {
    llama_set_input_rows_cb(ctx, client_rows_cb, this);
    llama_set_memory_op_cb(ctx, client_memory_cb, this);
}

// ---- the stage server --------------------------------------------------------------------------------------------

static std::vector<uint8_t> answer_error(const std::string & message) {
    std::vector<uint8_t> out{KIND_BYTES, 1};
    out.insert(out.end(), message.begin(), message.end());
    return out;
}

static std::vector<uint8_t> answer_ok(const std::vector<uint8_t> & data = {}) {
    std::vector<uint8_t> out{KIND_BYTES, 0};
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

struct server_state {
    llama_context * ctx;
    std::string token;
    int32_t n_batch;
    int32_t width;
    std::mutex mu;   // one model: the connections take turns
    std::atomic<bool> quit{false};
};

static std::vector<uint8_t> serve_rows(server_state & st, const std::vector<uint8_t> & req) {
    size_t at = 2;
    int64_t pos0 = 0;
    uint32_t n = 0;
    if (!get(req, at, pos0) || !get(req, at, n) || n == 0 || at + (size_t) n * 12 > req.size()) {
        return answer_error("a malformed rows request");
    }
    std::vector<int32_t> tok(n), pos(n), seq(n);
    std::memcpy(tok.data(), req.data() + at, n * 4); at += n * 4;
    std::memcpy(pos.data(), req.data() + at, n * 4); at += n * 4;
    std::memcpy(seq.data(), req.data() + at, n * 4);

    std::vector<uint8_t> out{KIND_TENSOR};
    put<uint32_t>(out, 1);
    out.push_back(0);
    out.reserve(out.size() + (size_t) n * st.width * sizeof(float));

    std::lock_guard<std::mutex> lock(st.mu);
    const int32_t step = std::max<int32_t>(1, st.n_batch);
    llama_batch batch = llama_batch_init(step, 0, 1);
    for (uint32_t i0 = 0; i0 < n; i0 += (uint32_t) step) {
        const int32_t m = (int32_t) std::min<uint32_t>((uint32_t) step, n - i0);
        batch.n_tokens = m;
        for (int32_t i = 0; i < m; ++i) {
            batch.token[i]     = tok[i0 + i];
            batch.pos[i]       = pos[i0 + i];
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seq[i0 + i];
            batch.logits[i]    = i == m - 1;   // the rows come for every token (llama_set_embeddings_nextn, unmasked)
        }
        if (llama_decode(st.ctx, batch) != 0) {
            llama_batch_free(batch);
            return answer_error("the decode failed");
        }
        for (int32_t i = 0; i < m; ++i) {
            const float * row = llama_get_embeddings_nextn_ith(st.ctx, i);
            if (!row) {
                llama_batch_free(batch);
                return answer_error("no hidden-state row for a token");
            }
            const auto * p = reinterpret_cast<const uint8_t *>(row);
            out.insert(out.end(), p, p + (size_t) st.width * sizeof(float));
        }
    }
    llama_batch_free(batch);
    return out;
}

static std::vector<uint8_t> serve_memory(server_state & st, const std::vector<uint8_t> & req) {
    if (req.size() < 3) {
        return answer_error("a malformed memory request");
    }
    const auto op = (llama_memory_op) req[2];
    size_t at = 3;
    int32_t seq = 0, seq_dst = 0, p0 = 0, p1 = 0, shift = 0;
    if (!get(req, at, seq) || !get(req, at, seq_dst) || !get(req, at, p0) || !get(req, at, p1) || !get(req, at, shift)) {
        return answer_error("a malformed memory request");
    }
    std::lock_guard<std::mutex> lock(st.mu);
    llama_memory_t mem = llama_get_memory(st.ctx);
    bool ok = true;
    switch (op) {
        case LLAMA_MEMORY_OP_CLEAR:    llama_memory_clear(mem, seq != 0); break;
        case LLAMA_MEMORY_OP_SEQ_RM:   ok = llama_memory_seq_rm(mem, seq, p0, p1); break;
        case LLAMA_MEMORY_OP_SEQ_CP:   llama_memory_seq_cp(mem, seq, seq_dst, p0, p1); break;
        case LLAMA_MEMORY_OP_SEQ_KEEP: llama_memory_seq_keep(mem, seq); break;
        case LLAMA_MEMORY_OP_SEQ_ADD:  llama_memory_seq_add(mem, seq, p0, p1, shift); break;
        case LLAMA_MEMORY_OP_SEQ_DIV:  llama_memory_seq_div(mem, seq, p0, p1, shift); break;
        default: return answer_error("an unknown memory operation");
    }
    return answer_ok({(uint8_t) (ok ? 1 : 0)});
}

static void serve_connection(server_state & st, int fd) {
    std::vector<uint8_t> req;
    bool hello = false;
    while (!st.quit && read_frame(fd, req)) {
        if (req.size() < 2 || req[0] != KIND_BYTES) {
            write_frame(fd, answer_error("a request is a bytes frame"));
            break;
        }
        const uint8_t op = req[1];
        if (!hello) {
            if (op != 'H' || !same_secret(std::string(req.begin() + 2, req.end()), st.token)) {
                write_frame(fd, answer_error("unauthorized"));
                break;
            }
            hello = true;
            std::vector<uint8_t> data;
            put<uint32_t>(data, (uint32_t) st.width);
            if (!write_frame(fd, answer_ok(data))) {
                break;
            }
            continue;
        }
        std::vector<uint8_t> answer;
        switch (op) {
            case 'P':
            case 'W': answer = serve_rows(st, req); break;
            case 'M': answer = serve_memory(st, req); break;
            case 'Q': st.quit = true; answer = answer_ok(); break;
            default:  answer = answer_error("an unknown request"); break;
        }
        if (!write_frame(fd, answer)) {
            break;
        }
    }
    close(fd);
}

int serve(llama_context * ctx, const std::string & addr, const std::string & token, int32_t n_batch) {
    std::string err;
    int port = 0;
    const int lfd = tcp_open(addr, true, err, &port);
    if (lfd < 0) {
        LOG_ERR("stage server: %s\n", err.c_str());
        return 1;
    }
    llama_set_embeddings_nextn(ctx, true, /*masked =*/ false);
    server_state st{ctx, token, n_batch, llama_input_rows_width(ctx), {}, {}};
    // the line a supervisor waits for
    LOG_INF("stage server: listening on port %d, rows of %d floats\n", port, st.width);
    while (!st.quit) {
        const int fd = accept(lfd, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        const int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::thread([&st, fd] { serve_connection(st, fd); }).detach();
    }
    close(lfd);
    return 0;
}

} // namespace stage_link

#endif
