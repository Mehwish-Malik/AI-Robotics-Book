
import React, { useEffect } from 'react';
import { useHistory } from '@docusaurus/router';

export default function Home() {
  const history = useHistory();

  useEffect(() => {
    // Redirect to first docs page
    history.replace('/docs/intro');
  }, [history]);

  return null; // Nothing renders on homepage
}

