#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 编译指令如下
// mpic++ -O2 main.cpp train.cpp guessing.cpp md5.cpp -o main

int main(int argc, char** argv)
{
    // MPI初始化
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 所有进程都需要执行这些初始化
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();

    if (rank == 0) {
        // 主进程处理输出和计时
        cout << "here" << endl;
        int curr_num = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        //auto start = system_clock::now();

        // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
        int history = 0;
        // std::ofstream a("./output/results.txt");
        while (!q.priority.empty())
        {
            // 决定一次处理多少个PT
            int num_pts_to_process = min(size, (int)q.priority.size());
            q.ProcessMultiplePTs(num_pts_to_process);
            q.total_guesses = q.guesses.size();
            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " <<history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                // 在此处更改实验生成的猜测上限
                int generate_n=10000000;
                if (history + q.total_guesses > 10000000)
                {
                    for (int i = 1; i < size; i++) {
                        int exit_signal = -1;
                        MPI_Send(&exit_signal, 1, MPI_INT, i, 999, MPI_COMM_WORLD);
                    }
                    double end = MPI_Wtime();
                    time_guess = end - start - time_hash;
                    cout << "Guess time:" << time_guess << "seconds"<< endl;
                    cout << "Hash time:" << time_hash << "seconds"<<endl;
                    cout << "Train time:" << time_train <<"seconds"<<endl;
                    break;
                }
            }
            // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
            // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
            if (curr_num > 1000000)
            {
                double start_hash = MPI_Wtime();
                bit32 state[4];
                for (string pw : q.guesses)
                {
                    // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                    MD5Hash(pw, state);

                    // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                    // a<<pw<<"\t";
                    // for (int i1 = 0; i1 < 4; i1 += 1)
                    // {
                    //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                    // }
                    // a << endl;
                }

                // 在这里对哈希所需的总时长进行计算
                double end_hash = MPI_Wtime();
                time_hash += (end_hash - start_hash);

                // 记录已经生成的口令总数
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }
        
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        // 其他进程执行相同的PopNext循环，但不处理输出
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
        }
        q.ProcessMultiplePTs(0);
    }
    
    MPI_Finalize();
    return 0;
}