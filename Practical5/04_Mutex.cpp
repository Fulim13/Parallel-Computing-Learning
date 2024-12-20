#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

std::mutex mtx; // mutex for critical section

void print_block(int n, char c)
{
    // critical section (exclusive access to std::cout signaled by locking mtx):
    mtx.lock();
    for (int i = 0; i < n; ++i)
    {
        std::cout << c;
    }
    std::cout << '\n';
    mtx.unlock();
}

int main()
{
    std::thread th1(print_block, 10000, '*');
    std::thread th2(print_block, 10000, '$');

    th1.join();
    th2.join();

    return 0;
}
