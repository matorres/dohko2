# Docko 2.0

**Docko 2.0** is a tool designed for generating black-box test cases for networking devices with graphical user interfaces (GUIs). Leveraging the power of reinforcement learning, Docko 2.0 dynamically explores and interacts with the GUI to identify different actions on run time, ensuring robust testing coverage.

## Features

- **Black-box Testing:** Generates test cases without needing detailed knowledge of the internal workings of the device. [In Progress]
- **Reinforcement Learning:** Uses RL algorithms to intelligently explore the GUI and discover new actions.
- **Dynamic Action Discovery:** Automatically scans the UI to identify and interact with available actions.
- **Scalable and Configurable:** Easily adaptable to different devices and GUI configurations.

## Installation

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)
- Google Chrome

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Generator:**
   Execute the following command to start generating test cases:

   ```bash
   python main.py --config config.yaml
   ```

## Configuration

The `config.yaml` file should include the device goal configuration. It should be written using the same syntax of the device config file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

