def get_roadmap(career):
    roadmaps = {
        "software engineer": [
            "Learn Python / Java",
            "Data Structures & Algorithms",
            "Build projects (apps, APIs)",
            "Internship / Open-source",
            "Apply for software roles"
        ],
        "data scientist": [
            "Learn Python & SQL",
            "Statistics & Probability",
            "Machine Learning basics",
            "Projects with real datasets",
            "Apply for Data Scientist roles"
        ],
        "web developer": [
            "HTML, CSS, JavaScript",
            "Frontend framework (React)",
            "Backend basics",
            "Build full-stack projects",
            "Apply for Web Developer roles"
        ]
    }

    return roadmaps.get(career.lower(), [
        "Learn core skills",
        "Build projects",
        "Gain experience",
        "Apply for jobs"
    ])
