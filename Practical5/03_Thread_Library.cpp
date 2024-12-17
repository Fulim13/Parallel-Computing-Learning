// thread example
#include <iostream> // std::cout
#include <thread>   // std::thread
// #include <windows.h> // Sleep() function is only available in Window
#include <chrono> // Protable, can use in both window and linux environment

void foo()
{
    // do stuff...
    // Sleep only available in Window (Because from window.h)
    // Sleep(4000);
    std::chrono::milliseconds timespan(4000);
    std::this_thread::sleep_for(timespan);
}

void bar(int x)
{
    // do stuff...
    // Sleep(5000);
    std::chrono::milliseconds timespan(5000);
    std::this_thread::sleep_for(timespan);
}

int main()
{
    std::thread first(foo);     // spawn new thread that calls foo()
    std::thread second(bar, 0); // spawn new thread that calls bar(0)

    std::cout << "main, foo and bar now execute concurrently...\n";

    // synchronize threads:
    first.join();  // pauses until first finishes
    second.join(); // pauses until second finishes

    std::cout << "foo and bar completed.\n";

    return 0;
}
