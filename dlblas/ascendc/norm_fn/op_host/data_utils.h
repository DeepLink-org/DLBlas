#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

/**
 * @brief Read data from file
 */
inline bool ReadFile(const std::string &filePath, size_t bufferSize, void *buffer, size_t bufferLen)
{
    if (buffer == nullptr) {
        ERROR_LOG("buffer is nullptr");
        return false;
    }
    if (bufferSize > bufferLen) {
        ERROR_LOG("buffer size is larger than buffer length");
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize != bufferSize) {
        ERROR_LOG("file size %zu != expected size %zu", fileSize, bufferSize);
        file.close();
        return false;
    }

    file.read(static_cast<char *>(buffer), bufferSize);
    if (!file) {
        ERROR_LOG("Read file failed");
        file.close();
        return false;
    }

    file.close();
    return true;
}

/**
 * @brief Write data to file
 */
inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    ssize_t writeSize = write(fd, buffer, size);
    close(fd);
    if (static_cast<size_t>(writeSize) != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

#endif
