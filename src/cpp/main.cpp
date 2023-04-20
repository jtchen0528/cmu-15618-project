#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::vector<double>> parse(std::string filename) 
{
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Error opening file");
    }

    std::string line{};
    std::vector<std::vector<double>> vvd{};
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> values{};
        std::string token{};
        while (std::getline(iss, token, ',')) {
            double value{0};
            try {
                value = std::stod(token);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing token " << token << ": " << e.what() << "\n";
                continue;
            }
            values.push_back(value);
        }
        // Do something with the values vector here
        for (auto value : values) {
            std::cout << value << " ";
        }
        vvd.push_back(values);
        std::cout << "\n";
    }

    infile.close();
    return vvd;
}

int main() {
    auto metadata(parse("iris_dataset.csv"));
    return 0;
}
