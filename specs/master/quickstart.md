# Quickstart Guide: Humanoid AI Robotics Book

## Overview
This guide helps you get started with the Humanoid AI Robotics book content and documentation setup.

## Prerequisites
- Node.js (version 16 or higher)
- npm or yarn package manager
- Git for version control
- Basic knowledge of Markdown and MDX

## Setup Documentation Environment

### 1. Clone the Repository
```bash
git clone <repository-url>
cd humanoid-robotics-book
```

### 2. Install Dependencies
```bash
npm install
# or
yarn install
```

### 3. Start Local Development Server
```bash
npm run start
# or
yarn start
```

This command starts a local development server and opens the documentation in your browser at `http://localhost:3000`.

### 4. Build for Production
```bash
npm run build
# or
yarn build
```

This generates static content in the `build` directory, which can be served using any static hosting service.

## Book Structure Navigation

The book is organized into 5 main modules:

1. **Foundation & Principles** - Core concepts and history
2. **Hardware Systems** - Actuators, sensors, and mechanical design
3. **Control Systems** - Motion planning and control algorithms
4. **AI & Perception** - Machine learning and computer vision
5. **Applications & Integration** - Real-world deployment and case studies

## Contributing Content

### Adding a New Chapter
1. Create a new Markdown file in the appropriate module directory
2. Add the chapter to the sidebar configuration
3. Include frontmatter with metadata:

```markdown
---
title: Chapter Title
description: Brief description of the chapter content
tags: [tag1, tag2]
sidebar_position: 3
---

# Chapter Title

Content here...
```

### Adding Code Examples
Use the following syntax for code examples with syntax highlighting:

```jsx
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="python" label="Python">

```python
# Python code example
def robot_control_loop():
    while True:
        sensor_data = read_sensors()
        motor_commands = process_control_logic(sensor_data)
        send_commands(motor_commands)
```

</TabItem>
<TabItem value="cpp" label="C++">

```cpp
// C++ code example
void RobotController::controlLoop() {
    while(running) {
        auto sensorData = readSensors();
        auto motorCommands = processControlLogic(sensorData);
        sendCommands(motorCommands);
    }
}
```

</TabItem>
</Tabs>
```

### Adding Diagrams
Place diagram files in the `static/img/diagrams/` directory and reference them:

```markdown
import useBaseUrl from '@docusaurus/useBaseUrl';

<img alt="Robot Control Architecture" src={useBaseUrl('/img/diagrams/control-architecture.svg')} />
```

## Running Tests
```bash
npm run test
# or
yarn test
```

This runs documentation validation including link checking and build verification.

## Deployment
The documentation can be deployed to various platforms:

- **GitHub Pages**: Use GitHub Actions workflow
- **Vercel**: Connect your GitHub repository
- **Netlify**: Drag and drop the build folder or connect GitHub
- **Self-hosted**: Serve the contents of the build folder with a web server

## Support
For issues with the documentation setup, create an issue in the repository or contact the maintainers.