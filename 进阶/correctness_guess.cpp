#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include<mpi.h>
#include <unordered_set>
using namespace std;
using namespace chrono;

// 编译指令如下
// mpic++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o correctness_guess
// mpic++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o correctness_guess -O1
// mpic++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o correctness_guess -O2

int main(int argc, char** argv)
{
    // MPI初始化
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 所有进程都需要执行这些初始化，因为Generate函数需要访问模型数据
    double time_hash = 0;
    double time_guess = 0; 
    double time_train = 0;
    PriorityQueue q;
    
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();

    if (rank == 0) {
        // 只有主进程处理测试数据和输出
        unordered_set<std::string> test_set;
        ifstream test_data("./input/Rockyou-singleLined-full.txt");
        int test_count = 0;
        string pw;
        while(test_data >> pw) {   
            test_count += 1;
            test_set.insert(pw);
            if (test_count >= 1000000) {
                break;
            }
        }
        int cracked = 0;

        cout << "Testing correctness with " << size << " processes" << endl;
        int curr_num = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        int history = 0;
        
        while (!q.priority.empty()) {
            // ===== 主要更改：使用ProcessMultiplePTs替换PopNext =====
            int num_pts_to_process = min(size, (int)q.priority.size());
            q.ProcessMultiplePTs(num_pts_to_process);
            // =============================================
            
            q.total_guesses = q.guesses.size();
            
            if (q.total_guesses - curr_num >= 100000) {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                if (history + q.total_guesses > 10000000) {
                    // 发送退出信号给所有工作进程
                    for (int i = 1; i < size; i++) {
                        int exit_signal = -1;
                        MPI_Send(&exit_signal, 1, MPI_INT, i, 999, MPI_COMM_WORLD);
                    }
                    
                    double end = MPI_Wtime();
                    time_guess = end - start - time_hash;
                    cout << "Guess time:" << time_guess << " seconds" << endl;
                    cout << "Hash time:" << time_hash << " seconds" << endl;
                    cout << "Train time:" << time_train << " seconds" << endl;
                    cout << "Cracked:" << cracked << " out of 1000000 passwords" << endl;
                    cout << "Success rate: " << (double)cracked / 1000000 * 100 << "%" << endl;
                    break;
                }
            }
            
            if (curr_num > 1000000) {
                double start_hash = MPI_Wtime();
                bit32 state[4];
                for (string pw : q.guesses) {
                    if (test_set.find(pw) != test_set.end()) {
                        cracked += 1;
                    }
                    MD5Hash(pw, state);
                }

                double end_hash = MPI_Wtime();
                time_hash += (end_hash - start_hash);

                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }
        
        // 确保所有工作进程都收到退出信号
        for (int i = 1; i < size; i++) {
            int exit_signal = -1;
            MPI_Send(&exit_signal, 1, MPI_INT, i, 999, MPI_COMM_WORLD);
        }
        
    } else {
        // ===== 主要更改：工作进程的循环逻辑 =====
        MPI_Barrier(MPI_COMM_WORLD);
        
        while (true) {
            // 检查是否收到退出信号
            int exit_signal;
            int flag;
            MPI_Iprobe(0, 999, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            if (flag) {
                MPI_Recv(&exit_signal, 1, MPI_INT, 0, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (exit_signal == -1) {
                    break;
                }
            }
            
            // 参与多PT处理（工作进程不需要指定PT数量）
            q.ProcessMultiplePTs(0);
        }
        // ========================================
    }
    
    MPI_Finalize();
    return 0;
}