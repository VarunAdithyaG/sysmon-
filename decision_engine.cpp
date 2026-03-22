#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;

struct Record {
    string timestamp;
    double cpu_percent{};
    double gpu_percent{};
    double memory_percent{};
    long long mem_used_kb{};
    long long mem_total_kb{};
    double disk_percent{};
    double cpu_temp_c{};
    double gpu_temp_c{};
    vector<int> top_pids;
};

namespace {

constexpr int SEQ_LEN = 30;
constexpr double KILL_THRESHOLD_MEM_PCT = 80.0;

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

bool parse_double(const string &s, double &out) {
    if (s.empty()) {
        return false;
    }
    char *end = nullptr;
    out = strtod(s.c_str(), &end);
    return end != s.c_str();
}

bool parse_long_long(const string &s, long long &out) {
    if (s.empty()) {
        return false;
    }
    char *end = nullptr;
    out = strtoll(s.c_str(), &end, 10);
    return end != s.c_str();
}

vector<Record> read_last_records(const string &path, int count) {
    ifstream in(path);
    if (!in) {
        cerr << "Failed to open " << path << "\n";
        return {};
    }

    string line;
    if (!getline(in, line)) {
        return {};
    }

    vector<Record> all;
    while (getline(in, line)) {
        if (line.empty()) continue;
        auto cols = split(line, ',');
        if (cols.size() < 10) {
            continue;
        }
        Record r;
        r.timestamp = cols[0];
        parse_double(cols[1], r.cpu_percent);
        parse_double(cols[2], r.gpu_percent);
        parse_double(cols[3], r.memory_percent);
        parse_long_long(cols[4], r.mem_used_kb);
        parse_long_long(cols[5], r.mem_total_kb);
        parse_double(cols[6], r.disk_percent);
        parse_double(cols[7], r.cpu_temp_c);
        parse_double(cols[8], r.gpu_temp_c);
        auto pid_parts = split(cols[9], '|');
        for (const auto &p : pid_parts) {
            try {
                int pid = stoi(p);
                r.top_pids.push_back(pid);
            } catch (...) {
                continue;
            }
        }
        all.push_back(move(r));
    }

    if ((int)all.size() <= count) {
        return all;
    }
    return vector<Record>(all.end() - count, all.end());
}

pair<double, double> mean_std(const vector<double> &v) {
    if (v.empty()) return {0.0, 0.0};
    double sum = 0.0;
    for (double x : v) sum += x;
    double mean = sum / v.size();
    double var = 0.0;
    for (double x : v) {
        double d = x - mean;
        var += d * d;
    }
    var /= v.size();
    double std = sqrt(var);
    return {mean, std};
}

}

int main(int argc, char **argv) {
    const string csv_path = "usage.csv";
    bool dry_run = true;
    if (argc > 1) {
        string arg = argv[1];
        if (arg == "--kill" || arg == "-k") {
            dry_run = false;
        }
    }

    auto records = read_last_records(csv_path, SEQ_LEN);
    if (records.empty()) {
        cerr << "No records found in " << csv_path << "\n";
        return 1;
    }

    vector<double> cpu_vals;
    vector<double> mem_vals;
    cpu_vals.reserve(records.size());
    mem_vals.reserve(records.size());

    for (const auto &r : records) {
        cpu_vals.push_back(r.cpu_percent);
        mem_vals.push_back(r.memory_percent);
    }

    auto [cpu_mean, cpu_std] = mean_std(cpu_vals);
    auto [mem_mean, mem_std] = mean_std(mem_vals);

    const Record &cur = records.back();

    bool cpu_anom = cpu_std > 0.0 && cur.cpu_percent > cpu_mean + 2.0 * cpu_std;
    bool mem_anom = mem_std > 0.0 && cur.memory_percent > mem_mean + 2.0 * mem_std;
    bool anomaly = cpu_anom || mem_anom;

    int kill_pid = -1;
    if (anomaly && cur.memory_percent >= KILL_THRESHOLD_MEM_PCT && !cur.top_pids.empty()) {
        kill_pid = cur.top_pids.front();
        if (!dry_run && kill_pid > 1) {
            kill(kill_pid, SIGKILL);
        }
    }

    cout << fixed << setprecision(2);
    cout << "{\n";
    cout << "  \"timestamp\": \"" << cur.timestamp << "\",\n";
    cout << "  \"cpu_mean\": " << cpu_mean << ",\n";
    cout << "  \"cpu_std\": " << cpu_std << ",\n";
    cout << "  \"mem_mean\": " << mem_mean << ",\n";
    cout << "  \"mem_std\": " << mem_std << ",\n";
    cout << "  \"cpu_current\": " << cur.cpu_percent << ",\n";
    cout << "  \"mem_current\": " << cur.memory_percent << ",\n";
    cout << "  \"anomaly\": " << (anomaly ? "true" : "false") << ",\n";
    cout << "  \"dry_run\": " << (dry_run ? "true" : "false") << ",\n";
    cout << "  \"top_pids\": [";
    for (size_t i = 0; i < cur.top_pids.size(); ++i) {
        cout << cur.top_pids[i];
        if (i + 1 < cur.top_pids.size()) cout << ", ";
    }
    cout << "],\n";
    cout << "  \"kill_pid\": " << kill_pid << "\n";
    cout << "}\n";

    return 0;
}

