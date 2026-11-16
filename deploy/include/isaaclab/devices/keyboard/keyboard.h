#pragma once

#include <string>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <thread>


/**
 * @brief Maintain a keyboard reading thread.
 * And get the latest key value.
 */
class Keyboard
{
public:
  Keyboard()
  {
    _hasTerminal = tcgetattr(fileno(stdin), &_oldSettings) == 0;
    _newSettings = _oldSettings;
    _newSettings.c_lflag &= ~(ICANON | ECHO);

    _startKey();
  }

  ~Keyboard()
  {
    _pauseKey();
  }

  void update()
  {
    const auto current_key = key();
    if(current_key != _last_key)
    {
      on_pressed = !current_key.empty();
      on_released = current_key.empty();
    }
    else
    {
      on_pressed = false;
      on_released = false;
    }
    
    _last_key = current_key;
  }

  /**
   * @brief Get the current key value
   * 
   * @return std::string 
   */
  std::string key() const {
    std::lock_guard<std::mutex> lock(_keyMutex);
    return _key;
  }

  /**
   * @brief Get the String object from keyboard 
   * 
   * @param slogan Used to prompt the user for input
   * @return std::string 
   */
  std::string getString(std::string slogan)
  {
    // Stop reading keyboard value
    _pauseKey();

    std::string stringtemp;
    std::cout << slogan << std::endl;// prompt
    std::getline(std::cin, stringtemp);

    // Restart reading keyboard value
    _startKey();
    return stringtemp;
  }

  /**
   * flags; available after update()
   */
  bool on_pressed = false;
  bool on_released = false;

  private:
  std::atomic<bool> _running{false};
  bool _hasTerminal = false;
  mutable std::mutex _keyMutex;
  std::thread _readThread;

  // Bound every byte read, including incomplete escape sequences, so stopping
  // the reader never waits indefinitely for another keyboard character.
  bool _readChar(char& c)
  {
    if (!_running) return false;
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fileno(stdin), &fds);
    timeval timeout{0, 80000};
    if (select(fileno(stdin) + 1, &fds, nullptr, nullptr, &timeout) <= 0 || !_running)
      return false;
    if (read(fileno(stdin), &c, 1) == 1) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5)); // EOF/error: avoid spinning
    return false;
  }

  void _read()
  {
    char c;
    std::string value;
    if (_readChar(c)) {
      if (c != '\033') value = c;
      else if (_readChar(c) && c == '[' && _readChar(c)) {
        switch (c) {
          case 'A': value = "up"; break;
          case 'B': value = "down"; break;
          case 'C': value = "right"; break;
          case 'D': value = "left"; break;
        }
      }
    }
    std::lock_guard<std::mutex> lock(_keyMutex);
    _key = value;
  }

  /**
   * @brief Restore keyboard default settings.
   */
  void _pauseKey()
  {
    _running = false;
    if (_readThread.joinable()) _readThread.join();
    if (_hasTerminal) tcsetattr(fileno(stdin), TCSANOW, &_oldSettings);
  }

  /**
   * @brief Disable canonical mode and echoing of input characters.
   */
  void _startKey()
  {
    if (_hasTerminal) tcsetattr(fileno(stdin), TCSANOW, &_newSettings);
    _running = true;
    _readThread = std::thread([this] {
      while (_running) _read();
    });
  }

  std::string _key, _last_key;
  
  termios _oldSettings{}, _newSettings{};
};
