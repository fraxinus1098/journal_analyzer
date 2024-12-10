# Journal Analyzer

A Python library for analyzing personal journals using emotional pattern detection, embedding-based analysis, and interactive visualizations.

## Features
- Emotional pattern analysis using GPT-4 and embeddings
- Multi-modal visualization of emotional journeys
- Pattern clustering and trend detection
- Secure input processing and validation
- Interactive HTML report generation

## Installation
```bash
# TODO: Add installation instructions
pip install journal-analyzer
```

## Quick Start
```python
# TODO: Add quick start example
from journal_analyzer import JournalAnalyzer

analyzer = JournalAnalyzer()
analyzer.process_entries("path/to/journals")
analyzer.generate_visualizations()
```

## Security
This library implements robust input validation and prompt injection protection. See the security documentation for details.

## Requirements
- Python 3.8+
- OpenAI API key
- Required packages listed in requirements.txt

## Privacy
It is recommend to save your journal entries .json files under .gitignore for privacy

# Grading Guidelines for LLM Class
Final project
The class includes a final project that is completed individually by default. This project should build off the course content. The project is graded as follows:

Technical accomplishment: 50%
Correctness and code quality: 20%
Utility: 10%
Documentation: 10%
Presentation/report quality: 10%
Graduate students may elect to complete the project in teams of two or three. Students electing so will have their final project graded in a different fashion. Students “technical accomplishment” will be graded with greater expectations.

The following acts or features are broadly indicative of technical accomplishment

Hosting a final project online so that it is accessible over the web in a way that is not a Jupyter notebook or similar. This will usually be a web application.
1. Training models
2. Fine-tuning models
3. Using retrieval augmented generation
4. Using multiple APIs or data sources
5. Using multiple models
6. Using multiple media (e.g. images or audio with large language models)
7. Using multiple calls to the same API
8. Resistance to prompt injection attacks
9. Utility is roughly speaking the usefulness of your project to humanity: the degree to which there exist living breathing humans who have a problem solved by your project and the degree to which those people might have the ability and willingness to pay for your solution. Of course, “pay” in this sense might be “publicly fund” if you’re working on a project with social benefit.

Correctness is roughly speaking the degree to which your code is both logically and mathematically correct and without errors. Does it do the thing you set out to do in the best way possible? Have you handled edge cases? Code quality includes standard indicia such as unit tests, docstrings, type hints, small function sizes, sensible abstractions, and the clear use of version control.

Documentation includes both code and end user documentation.

Presentations should include the following: 

1. A brief written document describing the purpose and design of your project or app. This must include appropriate citations to both academic work and software used in the project.
2. A brief video accessible to instructors in which you demonstrate use of the application or project.
3. A brief in-class presentation.
Presentations will be graded on the degree to which they are clear and compelling. The presentation grade is independent from the other dimensions of the project grade. 

## 1. A brief written document describing the purpose and design of your project or app. This must include appropriate citations to both academic work and software used in the project.
1. See README.md above, see [masterplan.md](https://github.com/fraxinus1098/journal_analyzer/blob/main/masterplan.md) which is a WIP and needs to be updated, and see resources and more details below.

## 2. A brief video accessible to instructors in which you demonstrate use of the application or project.

## 3. A brief in-class presentation.

## Resources and other key documentation
- https://github.com/fraxinus1098/journal_analyzer/blob/main/masterplan.md - A bit outdated and will need to be updated
- https://github.com/fraxinus1098/journal_analyzer/blob/main/project-structure.md - Does not reflect latest changes. Needs to be updated

## Works Cited and References
**How to effectively code using AI**
1. Followed this workflow for refining the idea and coding [Claude 3.5 Crash Course for Developers: Code 10x Faster in 2024 Claude 3.5 artifacts](https://www.youtube.com/watch?v=fMa2zQIkQwM&t)
2. Claude Pro Version using Claude 3.5 Sonnet, Artifacts, and Project Knowledgebase

**Generative Agent Simulations of 1,000 People**
1. [AI can now create a replica of your personality A two-hour interview is enough to accurately capture your values and preferences, according to new research from Stanford and Google DeepMind.](https://www.technologyreview.com/2024/11/20/1107100/ai-can-now-create-a-replica-of-your-personality/)
2. [Research Paper](https://arxiv.org/pdf/2411.10109)

**ammonhaggerty's digital twin**
1. https://github.com/ammonhaggerty/my-digital-twin?tab=readme-ov-file

**Reid AI**
1. [Transcript: The Futurist Summit: The Age of AI: Inflection Points with Reid Hoffman](https://www.washingtonpost.com/washington-post-live/2024/06/13/futurist-summit-age-ai-inflection-points-with-reid-hoffman/)
2. [The digital twin baby boom](https://www.axios.com/2024/07/09/digital-twins-reid-hoffman-health)

## Future Features/Optimization
**Optimization**
1. Get rid of HDBSCAN-based pattern detection and just use GPT-4o-mini for everything
2. Figure out better and easier way to store journal entries in a secure and private way
3. Use database instead of .json
4. Better normalize data
5. Improve data html visualization
6. Figure out core emotions that need to be shown. Maybe use official research
7. Update documentation

**Future Features**
1. Create a Spotify-like wrap, but for journaling instead
2. Integrate Spotify API user data to understand if there's a correlation between what people listen and how throughout the years and if that is correlated with journal entries
3. Integrate Google Photo API --> translate key images to text description and integrate with journal entries to track patterns
4. Publish as an official website? Profit!?!?

**User Research and Product Discovert**
1. Publish V 2.0 on https://www.reddit.com/r/Journaling/ and get user feedback. Refine and iterate

## Ultimate Goal
1. For my **Intro to AI** class, I made a simple AI digital twin using 2 hours of interview data based on research.
2. For my **LLM** class (this class), I attempted to analyze my 6 years worth of journal entries

