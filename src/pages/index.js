import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import styles from "./index.module.css";

// Hero image (put one good image in static/img/)
const heroImage = "/img/ai.png";

export default function Home() {
  return (
    <Layout title="Humanoid Robotics AI" description="AI Robotics Book">
      <div className={styles.hero}>
        {/* Large hero image */}
        <img
          src={heroImage}
          alt="Humanoid Robotics AI Book"
          className={styles.heroImage}
        />

        <h1>Humanoid Robotics AI</h1>
        <p>Explore the world of Artificial Intelligence, Robotics, and Human-Machine Intelligence.</p>

        <Link className={styles.button} to="/docs/intro">
          Start Reading
        </Link>
      </div>
    </Layout>
  );
}
