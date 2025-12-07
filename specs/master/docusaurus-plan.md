# Docusaurus Documentation Plan: Humanoid AI Robotics Book

## Overview
Complete plan for implementing the Humanoid AI Robotics book as a Docusaurus documentation website with professional, modern design and technical richness.

## Docusaurus Configuration

### 1. Project Setup
```bash
npx create-docusaurus@latest humanoid-robotics-book classic
cd humanoid-robotics-book
```

### 2. Core Configuration (`docusaurus.config.js`)
```javascript
// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Humanoid AI Robotics',
  tagline: 'A Comprehensive Guide to Building Humanoid Robots',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-website-url.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  organizationName: 'your-organization',
  projectName: 'humanoid-robotics-book',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false, // Disable blog if not needed
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Humanoid Robotics',
        logo: {
          alt: 'Humanoid Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            href: 'https://github.com/your-username/your-repo',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Content',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/humanoid-robotics',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/humanoid-robotics',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-username/your-repo',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Humanoid Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'cpp', 'bash', 'json'],
      },
    }),
};

module.exports = config;
```

## Folder Structure

```
humanoid-robotics-book/
├── docs/
│   ├── intro.md
│   ├── getting-started/
│   │   ├── overview.md
│   │   ├── installation.md
│   │   └── quickstart.md
│   ├── modules/
│   │   ├── module-1-fundamentals/
│   │   │   ├── chapter-1-introduction.md
│   │   │   ├── chapter-2-history.md
│   │   │   └── chapter-3-basic-concepts.md
│   │   ├── module-2-hardware/
│   │   │   ├── chapter-4-actuators.md
│   │   │   ├── chapter-5-sensors.md
│   │   │   └── chapter-6-control-systems.md
│   │   ├── module-3-control/
│   │   │   ├── chapter-7-motion-planning.md
│   │   │   ├── chapter-8-locomotion.md
│   │   │   └── chapter-9-balance.md
│   │   ├── module-4-ai/
│   │   │   ├── chapter-10-perception.md
│   │   │   ├── chapter-11-learning.md
│   │   │   └── chapter-12-decision-making.md
│   │   └── module-5-applications/
│   │       ├── chapter-13-simulation.md
│   │       ├── chapter-14-real-world-deployment.md
│   │       └── chapter-15-future-directions.md
│   ├── tutorials/
│   │   ├── basic-control-examples/
│   │   ├── simulation-workflows/
│   │   └── deployment-guides/
│   ├── reference/
│   │   ├── api-reference/
│   │   ├── code-examples/
│   │   └── hardware-specs/
│   ├── case-studies/
│   │   ├── tesla-optimus.md
│   │   ├── boston-dynamics-atlas.md
│   │   └── figure-ai.md
│   └── assets/
│       ├── diagrams/
│       ├── images/
│       └── videos/
├── src/
│   ├── components/
│   │   ├── HomepageFeatures/
│   │   │   ├── index.js
│   │   │   └── styles.module.css
│   │   └── CodeBlock/
│   │       ├── index.js
│   │       └── styles.module.css
│   ├── css/
│   │   └── custom.css
│   └── pages/
│       ├── index.js
│       ├── index.module.css
│       └── markdown-page.md
├── static/
│   ├── img/
│   │   ├── logo.svg
│   │   ├── favicon.ico
│   │   └── diagrams/
│   └── files/
│       └── code-examples/
├── docusaurus.config.js
├── package.json
├── sidebars.js
├── README.md
└── yarn.lock (or package-lock.json)
```

## MDX Layout Components

### 1. Custom Code Block Component
Create `src/components/CodeBlock/index.js`:
```jsx
import React from 'react';
import CodeBlock from '@theme/CodeBlock';

export default function CustomCodeBlock({children, ...props}) {
  return (
    <div className="code-block-wrapper">
      <CodeBlock {...props}>
        {children}
      </CodeBlock>
    </div>
  );
}
```

### 2. Diagram Component
Create `src/components/Diagram/index.js`:
```jsx
import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function Diagram({src, alt, caption}) {
  return (
    <div className="diagram-container">
      <img
        src={useBaseUrl(src)}
        alt={alt}
        className="diagram-image"
      />
      {caption && <p className="diagram-caption">{caption}</p>}
    </div>
  );
}
```

## Sidebar Configuration (`sidebars.js`)

```javascript
// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'autogenerated',
      dirName: '.',
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/overview',
        'getting-started/installation',
        'getting-started/quickstart'
      ],
    },
    {
      type: 'category',
      label: 'Fundamentals',
      items: [
        'modules/module-1-fundamentals/chapter-1-introduction',
        'modules/module-1-fundamentals/chapter-2-history',
        'modules/module-1-fundamentals/chapter-3-basic-concepts'
      ],
    },
    {
      type: 'category',
      label: 'Hardware Systems',
      items: [
        'modules/module-2-hardware/chapter-4-actuators',
        'modules/module-2-hardware/chapter-5-sensors',
        'modules/module-2-hardware/chapter-6-control-systems'
      ],
    },
    {
      type: 'category',
      label: 'Control Systems',
      items: [
        'modules/module-3-control/chapter-7-motion-planning',
        'modules/module-3-control/chapter-8-locomotion',
        'modules/module-3-control/chapter-9-balance'
      ],
    },
    {
      type: 'category',
      label: 'AI & Perception',
      items: [
        'modules/module-4-ai/chapter-10-perception',
        'modules/module-4-ai/chapter-11-learning',
        'modules/module-4-ai/chapter-12-decision-making'
      ],
    },
    {
      type: 'category',
      label: 'Applications',
      items: [
        'modules/module-5-applications/chapter-13-simulation',
        'modules/module-5-applications/chapter-14-real-world-deployment',
        'modules/module-5-applications/chapter-15-future-directions'
      ],
    },
    {
      type: 'category',
      label: 'Case Studies',
      items: [
        'case-studies/tesla-optimus',
        'case-studies/boston-dynamics-atlas',
        'case-studies/figure-ai'
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/basic-control-examples/index',
        'tutorials/simulation-workflows/index',
        'tutorials/deployment-guides/index'
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/api-reference/index',
        'reference/code-examples/index',
        'reference/hardware-specs/index'
      ],
    }
  ],
};

module.exports = sidebars;
```

## Deployment Strategy

### 1. GitHub Pages Deployment
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  deploy:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: npm

      - name: Install dependencies
        run: npm ci
      - name: Build website
        run: npm run build

      # Popular action to deploy to GitHub Pages:
      # Docs: https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-docusaurus
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          # Build output to publish to the `gh-pages` branch:
          publish_dir: ./build
          # The following lines assign commit authorship to the official
          # GH-Actions bot for deploys to `gh-pages` branch:
          # https://github.com/actions/checkout/issues/13#issuecomment-724415212
          # The GH actions bot is used by default if you didn't specify these params
          # commit_message: Deploy to GitHub Pages
          # committer: github-actions[bot] <github-actions[bot]@users.noreply.github.com>
          # author: ${{ github.actor }} <${{ github.actor }}@users.noreply.github.com>
```

### 2. Vercel Deployment
Create `vercel.json`:
```json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

### 3. Netlify Deployment
Create `netlify.toml`:
```toml
[build]
  command = "npm run build"
  publish = "build"
  environment = { NODE_VERSION = "18" }

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

## Package.json Scripts
```json
{
  "name": "humanoid-robotics-book",
  "version": "0.0.0",
  "private": true,
  "scripts": {
    "docusaurus": "docusaurus",
    "start": "docusaurus start",
    "build": "docusaurus build",
    "swizzle": "docusaurus swizzle",
    "deploy": "docusaurus deploy",
    "clear": "docusaurus clear",
    "serve": "docusaurus serve",
    "write-translations": "docusaurus write-translations",
    "write-heading-ids": "docusaurus write-heading-ids"
  },
  "dependencies": {
    "@docusaurus/core": "3.1.0",
    "@docusaurus/preset-classic": "3.1.0",
    "@mdx-js/react": "^3.0.0",
    "clsx": "^2.0.0",
    "prism-react-renderer": "^2.3.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@docusaurus/module-type-aliases": "3.1.0",
    "@docusaurus/types": "3.1.0"
  },
  "browserslist": {
    "production": [
      ">0.5%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 3 chrome version",
      "last 3 firefox version",
      "last 5 safari version"
    ]
  },
  "engines": {
    "node": ">=18.0"
  }
}
```

## Custom CSS (`src/css/custom.css`)
```css
/**
 * Custom CSS for Humanoid Robotics Book
 */

/* You can override the default Docusaurus styling here. */
:root {
  --ifm-color-primary: #25c2a0;
  --ifm-color-primary-dark: rgb(33, 175, 144);
  --ifm-color-primary-darker: rgb(31, 165, 136);
  --ifm-color-primary-darkest: rgb(26, 136, 112);
  --ifm-color-primary-light: rgb(70, 203, 174);
  --ifm-color-primary-lighter: rgb(102, 212, 189);
  --ifm-color-primary-lightest: rgb(146, 224, 208);
  --ifm-code-font-size: 95%;
}

/* Diagram container styling */
.diagram-container {
  margin: 2rem 0;
  text-align: center;
}

.diagram-image {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.diagram-caption {
  margin-top: 0.5rem;
  font-style: italic;
  color: #666;
  font-size: 0.9rem;
}

/* Code block enhancements */
.code-block-wrapper {
  margin: 1rem 0;
}

/* Custom admonitions for robotics content */
.roboto-admonition {
  border-left: 4px solid #25c2a0;
  background-color: #f8f9fa;
  padding: 1rem;
  margin: 1.5rem 0;
  border-radius: 0 4px 4px 0;
}

.roboto-admonition-title {
  font-weight: bold;
  margin-bottom: 0.5rem;
}

/* Module and chapter navigation */
.navbar {
  background-color: #242526;
}

/* Sidebar enhancements */
.menu {
  font-size: 0.9rem;
}

/* Table styling for specifications */
.doc-markdown table {
  display: block;
  overflow-x: auto;
  white-space: nowrap;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .diagram-image {
    width: 100%;
  }
}
```

This Docusaurus plan provides a complete framework for developing the Humanoid AI Robotics book as a professional documentation website with proper structure, styling, and deployment options.