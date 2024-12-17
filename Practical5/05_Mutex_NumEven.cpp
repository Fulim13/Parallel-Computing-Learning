#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

std::mutex increment;

void countEven(const std::vector<int> &numbers,
               int &numEven)
{
    increment.lock();
    for (const auto n : numbers)
    {
        if (n % 2 == 0)
        {
            numEven++;
        }
    }
    increment.unlock();
}
void insert_numbers_into_vector(std::vector<int> &v, int start_val, int len)
{
    for (int i = start_val; i < start_val + len; i++)
        v.insert(v.end(), i);
}

int main()
{
    // std::vector<int> numbers1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    // std::vector<int> numbers2 = { 11, 12, 13, 14, 15, 16, 17, 18 };
    std::vector<int> numbers1;
    std::vector<int> numbers2;

    int n = 0;

    insert_numbers_into_vector(numbers1, 0, 10000);
    insert_numbers_into_vector(numbers2, 20000, 10000);

    std::thread t1(countEven, std::ref(numbers1), std::ref(n));
    std::thread t2(countEven, std::ref(numbers2), std::ref(n));

    t1.join();
    t2.join();

    std::cout << "Total even numbers is: " << n << std::endl;
    return 0;
}
