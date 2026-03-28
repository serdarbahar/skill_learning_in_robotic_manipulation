# Skill Learning In Robotic Manipulation

Github repo for CMPE492

## Installation and Setup

#### Prerequisites
Ensure you have uv installed. If not, run:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Installation
Clone the repository and sync the environment.

```
git clone https://github.com/serdarbahar/skill_learning_in_robotic_manipulation.git
cd skill_learning_in_robotic_manipulation
uv sync
```

#### Environment Setup
Create a .env file in the root directory to manage your local paths:

```
echo 'DATA_PATH="/path/to/your/datasets"' > .env
```

#### Running the Code
Use uv run to execute scripts within the project environment:

```
uv run main.py
```
