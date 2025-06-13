#include "PCFG.h"
#include<mpi.h>
#include <sstream>
using namespace std;

void PriorityQueue::ProcessMultiplePTs(int num_pts) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        // 从优先队列中取出多个PT
        vector<PT> pts_to_process;
        int actual_pts = min(num_pts, (int)priority.size());
        
        // 从优先队列前端取出多个PT
        for (int i = 0; i < actual_pts; i++) {
            if (!priority.empty()) {
                pts_to_process.push_back(priority[i]);
            }
        }
        
        // 移除已取出的PT
        priority.erase(priority.begin(), priority.begin() + actual_pts);
        
        // 分配PT给各个进程
        int pts_per_process = actual_pts / size;
        int remaining_pts = actual_pts % size;
        
        // 主进程处理自己分配的PT
        int main_start = 0;
        int main_end = pts_per_process + (remaining_pts > 0 ? 1 : 0);
        
        for (int i = main_start; i < main_end && i < pts_to_process.size(); i++) {
            Generate(pts_to_process[i]);
        }
        
        // 向工作进程发送PT
        int current_pt_idx = main_end;
        for (int proc = 1; proc < size; proc++) {
            int proc_pts = pts_per_process + (proc <= remaining_pts ? 1 : 0);
            
            // 发送PT数量
            MPI_Send(&proc_pts, 1, MPI_INT, proc, 200, MPI_COMM_WORLD);
            
            // 发送每个PT的数据
            for (int j = 0; j < proc_pts && current_pt_idx < pts_to_process.size(); j++) {
                SendPTData(pts_to_process[current_pt_idx], proc);
                current_pt_idx++;
            }
        }
        
        // 接收工作进程处理的结果
        for (int proc = 1; proc < size; proc++) {
            int num_guesses;
            MPI_Recv(&num_guesses, 1, MPI_INT, proc, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (num_guesses > 0) {
                int data_size;
                MPI_Recv(&data_size, 1, MPI_INT, proc, 301, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                vector<char> recv_buffer(data_size);
                MPI_Recv(recv_buffer.data(), data_size, MPI_CHAR, proc, 302, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // 反序列化并添加到结果中
                string combined_strings(recv_buffer.begin(), recv_buffer.end());
                istringstream iss(combined_strings);
                string temp;
                while (getline(iss, temp, '\0')) {
                    if (!temp.empty()) {
                        guesses.emplace_back(temp);
                        total_guesses++;
                    }
                }
            }
        }
        
        // 为处理过的PT生成新的PT并插入队列
        for (PT& processed_pt : pts_to_process) {  // 改为非const引用
            vector<PT> new_pts = processed_pt.NewPTs();
            for (PT& new_pt : new_pts) {
                CalProb(new_pt);
                InsertPTToQueue(new_pt);
            }
        }
        
    } else {
        // 工作进程：接收并处理分配的PT
        int num_pts_to_process;
        MPI_Recv(&num_pts_to_process, 1, MPI_INT, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        vector<string> all_local_guesses;
        
        // 处理每个分配的PT
        for (int i = 0; i < num_pts_to_process; i++) {
            PT received_pt = ReceivePTData(0);
            
            // 保存当前guesses状态
            vector<string> old_guesses = guesses;
            int old_total = total_guesses;
            guesses.clear();
            total_guesses = 0;
            
            // 处理PT
            Generate(received_pt);
            
            // 收集结果
            all_local_guesses.insert(all_local_guesses.end(), guesses.begin(), guesses.end());
            
            // 恢复状态
            guesses = old_guesses;
            total_guesses = old_total;
        }
        
        // 发送结果回主进程
        int num_guesses = all_local_guesses.size();
        MPI_Send(&num_guesses, 1, MPI_INT, 0, 300, MPI_COMM_WORLD);
        
        if (num_guesses > 0) {
            // 序列化结果
            string serialized;
            for (const auto& guess : all_local_guesses) {
                serialized += guess + '\0';
            }
            
            int data_size = serialized.size();
            MPI_Send(&data_size, 1, MPI_INT, 0, 301, MPI_COMM_WORLD);
            MPI_Send(serialized.c_str(), data_size, MPI_CHAR, 0, 302, MPI_COMM_WORLD);
        }
    }
}

void PriorityQueue::SendPTData(const PT& pt, int dest_rank) {
    // 发送content size
    int content_size = pt.content.size();
    MPI_Send(&content_size, 1, MPI_INT, dest_rank, 210, MPI_COMM_WORLD);
    
    // 发送content数据
    for (const auto& seg : pt.content) {
        MPI_Send(&seg.type, 1, MPI_INT, dest_rank, 211, MPI_COMM_WORLD);
        MPI_Send(&seg.length, 1, MPI_INT, dest_rank, 212, MPI_COMM_WORLD);
    }
    
    // 发送max_indices
    int max_size = pt.max_indices.size();
    MPI_Send(&max_size, 1, MPI_INT, dest_rank, 213, MPI_COMM_WORLD);
    if (max_size > 0) {
        MPI_Send(pt.max_indices.data(), max_size, MPI_INT, dest_rank, 214, MPI_COMM_WORLD);
    }
    
    // 发送curr_indices
    int curr_size = pt.curr_indices.size();
    MPI_Send(&curr_size, 1, MPI_INT, dest_rank, 215, MPI_COMM_WORLD);
    if (curr_size > 0) {
        MPI_Send(pt.curr_indices.data(), curr_size, MPI_INT, dest_rank, 216, MPI_COMM_WORLD);
    }
    
    // 发送其他数据
    MPI_Send(&pt.pivot, 1, MPI_INT, dest_rank, 217, MPI_COMM_WORLD);
    MPI_Send(&pt.preterm_prob, 1, MPI_FLOAT, dest_rank, 218, MPI_COMM_WORLD);
    MPI_Send(&pt.prob, 1, MPI_FLOAT, dest_rank, 219, MPI_COMM_WORLD);
}

PT PriorityQueue::ReceivePTData(int source_rank) {
    PT pt;
    
    // 接收content size
    int content_size;
    MPI_Recv(&content_size, 1, MPI_INT, source_rank, 210, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 清空pt.content以避免重复数据
    pt.content.clear();
    for (int i = 0; i < content_size; i++) {
        int type, length;
        MPI_Recv(&type, 1, MPI_INT, source_rank, 211, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&length, 1, MPI_INT, source_rank, 212, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pt.content.emplace_back(type, length);  // 使用构造函数创建segment
    }
    
    // 接收max_indices
    int max_size;
    MPI_Recv(&max_size, 1, MPI_INT, source_rank, 213, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (max_size > 0) {
        pt.max_indices.resize(max_size);
        MPI_Recv(pt.max_indices.data(), max_size, MPI_INT, source_rank, 214, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // 接收curr_indices
    int curr_size;
    MPI_Recv(&curr_size, 1, MPI_INT, source_rank, 215, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (curr_size > 0) {
        pt.curr_indices.resize(curr_size);
        MPI_Recv(pt.curr_indices.data(), curr_size, MPI_INT, source_rank, 216, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // 接收其他数据
    MPI_Recv(&pt.pivot, 1, MPI_INT, source_rank, 217, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&pt.preterm_prob, 1, MPI_FLOAT, source_rank, 218, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&pt.prob, 1, MPI_FLOAT, source_rank, 219, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    return pt;
}

void PriorityQueue::InsertPTToQueue(const PT& pt) {
    // 如果队列为空，直接添加
    if (priority.empty()) {
        priority.emplace_back(pt);
        return;
    }
    
    // 按概率降序插入到合适位置
    bool inserted = false;
    for (auto iter = priority.begin(); iter != priority.end(); iter++) {
        // 对于非队首和队尾的特殊情况
        if (iter != priority.end() - 1 && iter != priority.begin()) {
            if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                priority.emplace(iter + 1, pt);
                inserted = true;
                break;
            }
        }
        // 如果是队尾位置
        if (iter == priority.end() - 1) {
            priority.emplace_back(pt);
            inserted = true;
            break;
        }
        // 如果概率比队首还高
        if (iter == priority.begin() && iter->prob < pt.prob) {
            priority.emplace(iter, pt);
            inserted = true;
            break;
        }
    }
    
    // 如果因为某种原因没有插入，添加到队尾
    if (!inserted) {
        priority.emplace_back(pt);
    }
}

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 获取MPI信息 /////////////////////////////////////
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的

        // 计算每个进程处理的范围
        int total_work = pt.max_indices[0];
        int work_per_process = total_work / size;
        int start = rank * work_per_process;
        int end = (rank == size - 1) ? total_work : start + work_per_process;

        // 每个进程处理自己的数据部分
        vector<string> local_guesses;
        local_guesses.reserve(work_per_process * 1.2);

        for (int i = start; i < end; i++) {
            local_guesses.emplace_back(a->ordered_values[i]);
        }

        // 收集所有进程的结果到rank 0
        if (rank == 0) {
            // 主进程收集数据
            guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
            total_guesses += local_guesses.size();
    
            // 接收其他进程的数据
            for (int i = 1; i < size; i++) {
                int recv_count;
                MPI_Recv(&recv_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
                if (recv_count > 0) {
                    vector<char> recv_buffer(recv_count);
                    MPI_Recv(recv_buffer.data(), recv_count, MPI_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
                    // 反序列化字符串
                    string combined_strings(recv_buffer.begin(), recv_buffer.end());
                    istringstream iss(combined_strings);
                    string temp;
                    while (getline(iss, temp, '\0')) {
                        if (!temp.empty()) {
                            guesses.emplace_back(temp);
                            total_guesses++;
                        }
                    }
                }
            }
        } else {
            // 其他进程发送数据到rank 0
            string serialized;
            for (const auto& str : local_guesses) {
                serialized += str + '\0';
            }
    
            int send_size = serialized.size();
            MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            if (send_size > 0) {
                MPI_Send(serialized.c_str(), send_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }
        /*
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
        */
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的

        // 计算每个进程处理的范围
        int total_work = pt.max_indices[pt.content.size() - 1];
        int work_per_process = total_work / size;
        int start = rank * work_per_process;
        int end = (rank == size - 1) ? total_work : start + work_per_process;

        // 每个进程处理自己的数据部分
        vector<string> local_guesses;
        local_guesses.reserve(work_per_process * 1.2);

        for (int i = start; i < end; i++) {
            string temp = guess + a->ordered_values[i];
            local_guesses.emplace_back(std::move(temp));
        }

        // 收集所有进程的结果到rank 0
        if (rank == 0) {
            // 主进程收集数据
            guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
            total_guesses += local_guesses.size();
    
            // 接收其他进程的数据
            for (int i = 1; i < size; i++) {
                int recv_count;
                MPI_Recv(&recv_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
                if (recv_count > 0) {
                    vector<char> recv_buffer(recv_count);
                    MPI_Recv(recv_buffer.data(), recv_count, MPI_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
                    // 反序列化字符串
                    string combined_strings(recv_buffer.begin(), recv_buffer.end());
                    istringstream iss(combined_strings);
                    string temp;
                    while (getline(iss, temp, '\0')) {
                        if (!temp.empty()) {
                            guesses.emplace_back(temp);
                            total_guesses++;
                        }
                   }
                }
            }
        } else {
            // 其他进程发送数据到rank 0
            string serialized;
            for (const auto& str : local_guesses) {
                serialized += str + '\0';
            }
    
            int send_size = serialized.size();
            MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            if (send_size > 0) {
                MPI_Send(serialized.c_str(), send_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }
        /*
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
        */
    }
}