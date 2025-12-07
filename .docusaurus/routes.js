import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '370'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '09f'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'c26'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'bdf'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '9c9'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '7d9'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', 'a14'),
    exact: true
  },
  {
    path: '/404',
    component: ComponentCreator('/404', 'da3'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'd2c'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '99a'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '08f'),
            routes: [
              {
                path: '/docs/assets/diagrams/placeholder',
                component: ComponentCreator('/docs/assets/diagrams/placeholder', '90f'),
                exact: true
              },
              {
                path: '/docs/assets/images/placeholder',
                component: ComponentCreator('/docs/assets/images/placeholder', '9aa'),
                exact: true
              },
              {
                path: '/docs/case-studies/boston-dynamics-atlas',
                component: ComponentCreator('/docs/case-studies/boston-dynamics-atlas', '5be'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/case-studies/figure-ai',
                component: ComponentCreator('/docs/case-studies/figure-ai', '3be'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/case-studies/tesla-optimus',
                component: ComponentCreator('/docs/case-studies/tesla-optimus', 'db8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/installation',
                component: ComponentCreator('/docs/getting-started/installation', '490'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/overview',
                component: ComponentCreator('/docs/getting-started/overview', '8dc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/quickstart',
                component: ComponentCreator('/docs/getting-started/quickstart', '4c6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', 'aed'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-1-fundamentals/chapter-1-introduction',
                component: ComponentCreator('/docs/modules/module-1-fundamentals/chapter-1-introduction', '7b6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-1-fundamentals/chapter-2-history',
                component: ComponentCreator('/docs/modules/module-1-fundamentals/chapter-2-history', '1bc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-1-fundamentals/chapter-3-basic-concepts',
                component: ComponentCreator('/docs/modules/module-1-fundamentals/chapter-3-basic-concepts', '45e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-2-hardware/chapter-4-actuators',
                component: ComponentCreator('/docs/modules/module-2-hardware/chapter-4-actuators', '3ac'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-2-hardware/chapter-5-sensors',
                component: ComponentCreator('/docs/modules/module-2-hardware/chapter-5-sensors', 'fdf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-2-hardware/chapter-6-control-systems',
                component: ComponentCreator('/docs/modules/module-2-hardware/chapter-6-control-systems', '9fc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-3-control/chapter-7-motion-planning',
                component: ComponentCreator('/docs/modules/module-3-control/chapter-7-motion-planning', '510'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-3-control/chapter-8-locomotion',
                component: ComponentCreator('/docs/modules/module-3-control/chapter-8-locomotion', '7a4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-3-control/chapter-9-balance',
                component: ComponentCreator('/docs/modules/module-3-control/chapter-9-balance', 'ed7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-4-ai/chapter-10-perception',
                component: ComponentCreator('/docs/modules/module-4-ai/chapter-10-perception', 'c54'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-4-ai/chapter-11-learning',
                component: ComponentCreator('/docs/modules/module-4-ai/chapter-11-learning', 'c4d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-4-ai/chapter-12-decision-making',
                component: ComponentCreator('/docs/modules/module-4-ai/chapter-12-decision-making', '006'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-5-applications/chapter-13-simulation',
                component: ComponentCreator('/docs/modules/module-5-applications/chapter-13-simulation', '74e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-5-applications/chapter-14-real-world-deployment',
                component: ComponentCreator('/docs/modules/module-5-applications/chapter-14-real-world-deployment', 'c8f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-5-applications/chapter-15-future-directions',
                component: ComponentCreator('/docs/modules/module-5-applications/chapter-15-future-directions', '32f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-5-applications/chapter-16-case-studies',
                component: ComponentCreator('/docs/modules/module-5-applications/chapter-16-case-studies', '2a2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/reference/api-reference/',
                component: ComponentCreator('/docs/reference/api-reference/', '6f9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/reference/code-examples/',
                component: ComponentCreator('/docs/reference/code-examples/', '215'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/reference/hardware-specs/',
                component: ComponentCreator('/docs/reference/hardware-specs/', 'b97'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorials/basic-control-examples/',
                component: ComponentCreator('/docs/tutorials/basic-control-examples/', 'c97'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorials/basic-control-examples/balance-control',
                component: ComponentCreator('/docs/tutorials/basic-control-examples/balance-control', 'bd4'),
                exact: true
              },
              {
                path: '/docs/tutorials/basic-control-examples/joint-control-basics',
                component: ComponentCreator('/docs/tutorials/basic-control-examples/joint-control-basics', '19f'),
                exact: true
              },
              {
                path: '/docs/tutorials/basic-control-examples/simple-locomotion',
                component: ComponentCreator('/docs/tutorials/basic-control-examples/simple-locomotion', 'dd1'),
                exact: true
              },
              {
                path: '/docs/tutorials/deployment-guides/',
                component: ComponentCreator('/docs/tutorials/deployment-guides/', '3fd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorials/simulation-workflows/',
                component: ComponentCreator('/docs/tutorials/simulation-workflows/', '4e7'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '876'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
