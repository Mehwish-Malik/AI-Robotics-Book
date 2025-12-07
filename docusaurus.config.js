// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Humanoid AI Robotics',
  tagline: 'A Comprehensive Guide to Building Humanoid Robots',
  favicon: 'img/favicon.ico',

  url: 'https://humanoid-robotics-book.com',
  baseUrl: '/',

  organizationName: 'Mehwish-Malik',
  projectName: 'AI-Robotics-Book',
  trailingSlash: false,
  deploymentBranch: 'gh-pages',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

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
          editUrl: 'https://github.com/Mehwish-Malik/AI-Robotics-Book/tree/main/',
          routeBasePath: 'docs',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: /** @type {import('@docusaurus/preset-classic').ThemeConfig} */ ({
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: '🚀 Humanoid Robotics', // Added emoji for fun and boldness
      hideOnScroll: true,           // Navbar hides on scroll
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: '📖 Book',
        },
        {
          href: 'https://github.com/humanoid-robotics/humanoid-robotics-book',
          label: 'GitHub',
          position: 'right',
          className: 'button button--primary', // makes it a button
        },
      ],
      style: 'dark',
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            { label: 'Introduction', to: '/docs/intro' },
            { label: 'Modules', to: '/docs/modules/module-1-fundamentals/chapter-1-introduction' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'Stack Overflow', href: 'https://stackoverflow.com/questions/tagged/humanoid-robotics' },
            { label: 'Robotics Stack Exchange', href: 'https://robotics.stackexchange.com/' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'GitHub', href: 'https://github.com/humanoid-robotics/humanoid-robotics-book' },
            { label: 'Research Papers', href: 'https://scholar.google.com/scholar?q=humanoid+robotics' },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Humanoid AI Robotics Book • Designed by Wish Malik`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python','cpp','bash','json','typescript','csharp','java'],
    },
  }),
};

module.exports = config;
