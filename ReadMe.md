# ModuSync PoC

A brief description of your project or application goes here.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.x
- pip (Python package installer)
- tmux (terminal multiplexer)

## Installation Instructions

### 1. Install `tmux`

You can install `tmux` using your system's package manager:

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tmux
```

#### On macOS (using Homebrew):
```bash
brew install tmux
```

### 2. Install Python Dependencies

Install any required Python packages using `pip`. Make sure you're in the project directory:

```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt` file, you can install individual packages as needed.

## Usage

To start the application, simply run:

```bash
./run_all.sh
```

This script will use `tmux` to manage multiple terminal sessions and launch the necessary components of your project.

> Ensure the script has executable permissions. If not, run:  
```bash
chmod +x run_all.sh
```

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

