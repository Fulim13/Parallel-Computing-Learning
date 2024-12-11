#include <iostream>
#include <omp.h>

using namespace std;

#define MAX_EMPLOYEE 10000000

enum class Co
{
    Company1,
    Company2
};

int fetchTheSalary(const int employee,
                   const Co company)
{
    if (company == Co::Company1)
    {
        return (employee < 100) ? 99340 : 54300;
    }
    else
    {
        return (employee < 1000) ? 88770 : 57330;
    }
}

int company_salaries_q1()
{
    int salaries1 = 0;
    int salaries2 = 0;
    double runtime;
    runtime = omp_get_wtime();
#pragma omp parallel shared(salaries1, salaries2) num_threads(16)
    {
#pragma omp for reduction(+ : salaries1)
        for (int employee = 0; employee < MAX_EMPLOYEE; employee++)
            salaries1 += fetchTheSalary(employee, Co::Company1);

#pragma omp critical
        {
            std::cout << "Salaries1: " << salaries1 << std::endl;
            // printf("Salaries1: %d\n", salaries1); // printf is thread safe
        }

#pragma omp for reduction(+ : salaries2)
        for (int employee = 0; employee < MAX_EMPLOYEE; employee++)
            salaries2 += fetchTheSalary(employee, Co::Company2);

#pragma omp critical
        {
            std::cout << "Salaries2: " << salaries2 << std::endl;
            // printf("Salaries2: %d\n", salaries2);
        }
    }
    runtime = omp_get_wtime() - runtime;
    std::cout << " In " << runtime << " seconds" << std::endl;
    return 0;
}

int company_salaries_q2()
{
    int salaries1 = 0;
    int salaries2 = 0;
    double runtime;
    runtime = omp_get_wtime();
#pragma omp parallel shared(salaries1, salaries2) num_threads(16)
    {
#pragma omp for reduction(+ : salaries1, salaries2)
        for (int employee = 0; employee < MAX_EMPLOYEE; employee++)
        {
            salaries1 += fetchTheSalary(employee, Co::Company1);
            salaries2 += fetchTheSalary(employee, Co::Company2);
        }
        std::cout << "Salaries1: " << salaries1 << std::endl;
        std::cout << "Salaries2: " << salaries2 << std::endl;
    }
    runtime = omp_get_wtime() - runtime;
    std::cout << " In " << runtime << " seconds" << std::endl;
    return 0;
}

int company_salaries_q3()
{
    int salaries1 = 0;
    int salaries2 = 0;
    double runtime;
    runtime = omp_get_wtime();
#pragma omp parallel shared(salaries1, salaries2) num_threads(16)
    {
#pragma omp for reduction(+ : salaries1, salaries2)
        for (int employee = 0; employee < MAX_EMPLOYEE; employee++)
        {
            salaries1 += fetchTheSalary(employee, Co::Company1);
            salaries2 += fetchTheSalary(employee, Co::Company2);
        }
#pragma omp single
        {
            std::cout << "Salaries1: " << salaries1 << std::endl;
            std::cout << "Salaries2: " << salaries2 << std::endl;
        }
    }
    runtime = omp_get_wtime() - runtime;
    std::cout << " In " << runtime << " seconds" << std::endl;
    return 0;
}

int main()
{
    // company_salaries_q1();
    company_salaries_q2();
    // company_salaries_q3();
    return 0;
}
