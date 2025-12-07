// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'intro',
        'getting-started/overview',
        'getting-started/installation',
        'getting-started/quickstart'
      ],
    },
    {
      type: 'category',
      label: 'Module 1: Foundation & Principles',
      items: [
        'modules/module-1-fundamentals/chapter-1-introduction',
        'modules/module-1-fundamentals/chapter-2-history',
        'modules/module-1-fundamentals/chapter-3-basic-concepts'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Hardware Systems',
      items: [
        'modules/module-2-hardware/chapter-4-actuators',
        'modules/module-2-hardware/chapter-5-sensors',
        'modules/module-2-hardware/chapter-6-control-systems'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Control Systems',
      items: [
        'modules/module-3-control/chapter-7-motion-planning',
        'modules/module-3-control/chapter-8-locomotion',
        'modules/module-3-control/chapter-9-balance'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: AI & Perception',
      items: [
        'modules/module-4-ai/chapter-10-perception',
        'modules/module-4-ai/chapter-11-learning',
        'modules/module-4-ai/chapter-12-decision-making'
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Applications & Integration',
      items: [
        'modules/module-5-applications/chapter-13-simulation',
        'modules/module-5-applications/chapter-14-real-world-deployment',
        'modules/module-5-applications/chapter-15-future-directions',
        'modules/module-5-applications/chapter-16-case-studies'
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