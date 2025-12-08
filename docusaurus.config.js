// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Humanoid AI Robotics',
  tagline: 'A Comprehensive Guide to Building Humanoid Robots',
  favicon: 'img/favicon.ico',

  // 🌐 Correct config for Vercel Deployment
  url: 'https://ai-robotics-book.vercel.app', // You can update this after deploy
  baseUrl: '/',
  organizationName: 'Mehwish-Malik',
  projectName: 'AI-Robotics-Book',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/Mehwish-Malik/AI-Robotics-Book/tree/main/',
          routeBasePath: 'docs',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',

    navbar: {
      title: '🚀 Humanoid Robotics',
      hideOnScroll: true,
      style: 'dark',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: '📖 Book',
        },
        {
          href: 'https://github.com/Mehwish-Malik/AI-Robotics-Book',
          label: 'GitHub',
          position: 'right',
          className: 'button button--primary',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            { label: 'Introduction', to: '/docs/intro' },
            {
              label: 'Modules',
              to: '/docs/modules/module-1-fundamentals/chapter-1-introduction',
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
              label: 'Robotics Stack Exchange',
              href: 'https://robotics.stackexchange.com/',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Mehwish-Malik/AI-Robotics-Book',
            },
            {
              label: 'Research Papers',
              href: 'https://scholar.google.com/scholar?q=humanoid+robotics',
            },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Humanoid AI Robotics Book • Designed by Wish Malik`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: [
        'python',
        'cpp',
        'bash',
        'json',
        'typescript',
        'csharp',
        'java',
      ],
    },
  },
};

export default config;
