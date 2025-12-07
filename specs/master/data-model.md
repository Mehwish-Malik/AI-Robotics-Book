# Data Model: Humanoid AI Robotics Book Blueprint

## Overview
This document defines the data structures and models for the Humanoid AI Robotics book content management system.

## Content Entities

### 1. Book Module
- **id**: string (unique identifier)
- **title**: string (module title)
- **description**: string (module description)
- **order**: integer (sequence number)
- **learningGoals**: array of strings (learning objectives)
- **prerequisites**: array of strings (required knowledge)
- **chapters**: array of Chapter references
- **caseStudies**: array of CaseStudy references
- **createdAt**: datetime
- **updatedAt**: datetime

### 2. Book Chapter
- **id**: string (unique identifier)
- **moduleId**: string (reference to parent module)
- **title**: string (chapter title)
- **description**: string (chapter summary)
- **order**: integer (sequence within module)
- **estimatedReadingTime**: integer (in minutes)
- **difficultyLevel**: enum ('beginner', 'intermediate', 'advanced')
- **learningObjectives**: array of strings (specific objectives)
- **keyTopics**: array of strings (main topics covered)
- **codeExamples**: array of CodeExample references
- **diagrams**: array of Diagram references
- **exercises**: array of Exercise references
- **references**: array of strings (external references)
- **createdAt**: datetime
- **updatedAt**: datetime

### 3. Code Example
- **id**: string (unique identifier)
- **chapterId**: string (reference to parent chapter)
- **title**: string (example title)
- **description**: string (what the example demonstrates)
- **language**: string (programming language)
- **code**: string (the actual code)
- **explanation**: string (step-by-step explanation)
- **useCase**: string (practical application)
- **createdAt**: datetime
- **updatedAt**: datetime

### 4. Diagram
- **id**: string (unique identifier)
- **chapterId**: string (reference to parent chapter)
- **title**: string (diagram title)
- **description**: string (what the diagram shows)
- **type**: enum ('architecture', 'flowchart', 'block', 'sequence', 'other')
- **sourceFile**: string (path to diagram file)
- **caption**: string (explanatory text)
- **altText**: string (accessibility text)
- **createdAt**: datetime
- **updatedAt**: datetime

### 5. Case Study
- **id**: string (unique identifier)
- **title**: string (case study title)
- **company**: string (company/organization)
- **robotName**: string (robot system name)
- **description**: string (overview of the case)
- **technicalApproach**: string (technical implementation details)
- **challenges**: array of strings (technical challenges faced)
- **solutions**: array of strings (solutions implemented)
- **results**: string (outcomes and performance)
- **lessonsLearned**: array of strings (key takeaways)
- **relatedModules**: array of Module references
- **createdAt**: datetime
- **updatedAt**: datetime

### 6. Exercise
- **id**: string (unique identifier)
- **chapterId**: string (reference to parent chapter)
- **title**: string (exercise title)
- **description**: string (what the exercise involves)
- **type**: enum ('theoretical', 'practical', 'coding', 'analysis')
- **difficulty**: enum ('beginner', 'intermediate', 'advanced')
- **instructions**: string (step-by-step instructions)
- **expectedOutcome**: string (what should be achieved)
- **hints**: array of strings (helpful tips)
- **solution**: string (solution approach)
- **createdAt**: datetime
- **updatedAt**: datetime

### 7. Tutorial
- **id**: string (unique identifier)
- **title**: string (tutorial title)
- **description**: string (tutorial overview)
- **estimatedTime**: integer (time to complete in minutes)
- **prerequisites**: array of strings (required knowledge/skills)
- **steps**: array of objects with properties:
  - **stepNumber**: integer
  - **title**: string
  - **description**: string
  - **code**: string (optional code example)
  - **diagram**: string (optional diagram reference)
- **relatedChapters**: array of Chapter references
- **createdAt**: datetime
- **updatedAt**: datetime

### 8. Reference Material
- **id**: string (unique identifier)
- **title**: string (reference title)
- **type**: enum ('api', 'specification', 'paper', 'book', 'article', 'video')
- **url**: string (link to resource)
- **description**: string (summary of content)
- **category**: string (classification)
- **relatedChapters**: array of Chapter references
- **createdAt**: datetime
- **updatedAt**: datetime

## Relationships

1. Book Module 1-to-many Book Chapters
2. Book Chapter 1-to-many Code Examples
3. Book Chapter 1-to-many Diagrams
4. Book Chapter 1-to-many Exercises
5. Book Module 1-to-many Case Studies (many-to-many through ModuleCaseStudy junction)
6. Tutorial 1-to-many Chapters (many-to-many through TutorialChapter junction)
7. Reference Material 1-to-many Chapters (many-to-many through ReferenceChapter junction)

## Validation Rules

1. Module title must be 3-100 characters
2. Chapter title must be 3-100 characters
3. Learning objectives must be 1-5 items per chapter
4. Difficulty level must be one of the defined enums
5. Order values must be unique within parent collections
6. Required fields cannot be null or empty
7. Code examples must have valid syntax highlighting language
8. Diagram types must be one of the defined enums