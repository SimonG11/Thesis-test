def get_ericsson_tasks():

    tasks = [
    {
        "id": 1,
        "task_name": "Pre-assesment or workshop preparation",
        "base_effort": 800,
        "min": 1,
        "max": 6,
        "dependencies": [],
        "resource": "SA/SD SolArch (JS8) - Solution Architect - DA"
    },
    {
        "id": 2,
        "task_name": "Collection of Requirements ( Requirements Gathering workshopts)",
        "base_effort": 581.2,
        "min": 1,
        "max": 6,
        "dependencies": [
            1
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 3,
        "task_name": "NFR Requirements ( Collection, Analysis, Dimensioning)",
        "base_effort": 108.8,
        "min": 1,
        "max": 2,
        "dependencies": [
            1
        ],
        "resource": "SDU SolArch (JS6) - DevOps Tool Admin"
    },
    {
        "id": 4,
        "task_name": "Collection of Performance Testing Requirement",
        "base_effort": 176,
        "min": 1,
        "max": 2,
        "dependencies": [
            1
        ],
        "resource": "SDU SolArch (JS6) - Performance Lead"
    },
    {
        "id": 5,
        "task_name": "Data Migration Strategy and Assessment",
        "base_effort": 160,
        "min": 1,
        "max": 2,
        "dependencies": [
            2
        ],
        "resource": "SDU CPM (JS7) - DM CPM"
    },
    {
        "id": 6,
        "task_name": "Env and Infra Specifications (HA, Addons, Timeplan)",
        "base_effort": 108.80000000000001,
        "min": 1,
        "max": 2,
        "dependencies": [
            2
        ],
        "resource": "SDU SolArch (JS6) - DevOps Tool Admin"
    },
    {
        "id": 7,
        "task_name": "Infrastructure GeoRedundancy and-or Disaster Recovery analysis",
        "base_effort": 335.36,
        "min": 1,
        "max": 2,
        "dependencies": [
            2
        ],
        "resource": "SDU SolArch (JS6) - DevOps Tool Admin"
    },
    {
        "id": 8,
        "task_name": "Remote Connectivity Planning",
        "base_effort": 80,
        "min": 1,
        "max": 2,
        "dependencies": [
            2
        ],
        "resource": "SDU SolArch (JS6) - DevOps Tool Admin"
    },
    {
        "id": 9,
        "task_name": "Customer Test Strategy and Specifications",
        "base_effort": 718.5684561728751,
        "min": 1,
        "max": 6,
        "dependencies": [
            2
        ],
        "resource": "MA TestMgr (JS6) - Test lead"
    },
    {
        "id": 10,
        "task_name": "Performance Test Strategy and Plan",
        "base_effort": 176,
        "min": 1,
        "max": 2,
        "dependencies": [
            2
        ],
        "resource": "SDU SolArch (JS6) - Performance Lead"
    },
    {
        "id": 11,
        "task_name": "HLD - PLM and ILM-Resource Management",
        "base_effort": 30.720000000000006,
        "min": 1,
        "max": 1,
        "dependencies": [
            10
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 12,
        "task_name": "HLD - Operational Support and Readiness",
        "base_effort": 39.519999999999996,
        "min": 1,
        "max": 1,
        "dependencies": [
            10
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 13,
        "task_name": "HLD - Revenue Management",
        "base_effort": 481.34400000000016,
        "min": 1,
        "max": 3,
        "dependencies": [
            10
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 14,
        "task_name": "HLD Integration Architecture",
        "base_effort": 64,
        "min": 1,
        "max": 2,
        "dependencies": [
            13
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 15,
        "task_name": "HLD Product Model - Catalog",
        "base_effort": 384.4,
        "min": 1,
        "max": 3,
        "dependencies": [
            57
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 16,
        "task_name": "HLD E2E Architecture - Common",
        "base_effort": 161.23439999999997,
        "min": 1,
        "max": 2,
        "dependencies": [
            13
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 17,
        "task_name": "HLD Infrastructure and IP Design",
        "base_effort": 340.48,
        "min": 1,
        "max": 2,
        "dependencies": [
            14
        ],
        "resource": "SDU SolArch (JS6) - DevOps Tool Admin"
    },
    {
        "id": 18,
        "task_name": "HLD Non-Functional Design",
        "base_effort": 283.584,
        "min": 1,
        "max": 2,
        "dependencies": [
            14
        ],
        "resource": "SDU ITSysExp (JS5) - SW SME"
    },
    {
        "id": 19,
        "task_name": "HLD CBEV Functional",
        "base_effort": 989.462,
        "min": 1,
        "max": 4,
        "dependencies": [
            13
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 20,
        "task_name": "Low Level Design E2E Common",
        "base_effort": 2850.1755294117647,
        "min": 1,
        "max": 6,
        "dependencies": [
            19
        ],
        "resource": "SDU IntegEngin (JS6) - CAF IE"
    },
    {
        "id": 21,
        "task_name": "Low Level Design Infrastructure",
        "base_effort": 470.4,
        "min": 1,
        "max": 3,
        "dependencies": [
            18
        ],
        "resource": "SDU SolArch (JS6) - DevOps Tool Admin"
    },
    {
        "id": 22,
        "task_name": "Low Level Design CBEV",
        "base_effort": 1114.8,
        "min": 1,
        "max": 6,
        "dependencies": [
            19
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 23,
        "task_name": "Alarms Guide and Counter Solution + Operations - Troubelshooting Guide",
        "base_effort": 320,
        "min": 1,
        "max": 6,
        "dependencies": [
            22
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 24,
        "task_name": "User Manuals - System Administration Guide",
        "base_effort": 160,
        "min": 1,
        "max": 3,
        "dependencies": [
            21
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 25,
        "task_name": "Deployment - System Configuration and Application UI",
        "base_effort": 160,
        "min": 1,
        "max": 6,
        "dependencies": [
            21
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 26,
        "task_name": "Low Level Design CBEV RMCA",
        "base_effort": 876,
        "min": 1,
        "max": 4,
        "dependencies": [
            25
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 27,
        "task_name": "Low Level Design - NFR",
        "base_effort": 197.76,
        "min": 1,
        "max": 3,
        "dependencies": [
            18
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 28,
        "task_name": "Site Survey, Deployment Physical Infra and Security",
        "base_effort": 278.4,
        "min": 1,
        "max": 3,
        "dependencies": [
            21
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 29,
        "task_name": "External or Additional elements installation",
        "base_effort": 40,
        "min": 1,
        "max": 1,
        "dependencies": [
            28
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 30,
        "task_name": "Infrastructure GeoRedundancy and-or Disaster Recovery activities and verification",
        "base_effort": 440,
        "min": 1,
        "max": 4,
        "dependencies": [
            29
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 31,
        "task_name": "Infrastructure Backup and Recovery deployment and verification",
        "base_effort": 288,
        "min": 1,
        "max": 6,
        "dependencies": [
            30
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 32,
        "task_name": "Infrastructure readiness verification, baby sitting and support",
        "base_effort": 2392,
        "min": 1,
        "max": 4,
        "dependencies": [
            29
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 33,
        "task_name": "Internal Environment Setup (Including DevOps infra activities)",
        "base_effort": 140,
        "min": 1,
        "max": 4,
        "dependencies": [
            29
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 34,
        "task_name": "Internal Handover",
        "base_effort": 192,
        "min": 1,
        "max": 4,
        "dependencies": [
            33
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 35,
        "task_name": "Platform Verification (Virtual or Containerized)",
        "base_effort": 28,
        "min": 1,
        "max": 1,
        "dependencies": [
            32
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 36,
        "task_name": "DevOps tool installation and SWDP Solution Initial Setup",
        "base_effort": 62.076,
        "min": 1,
        "max": 6,
        "dependencies": [
            35
        ],
        "resource": "SDU SolArch (JS6) - DevOps SME"
    },
    {
        "id": 37,
        "task_name": "Solution Delpoyment and Integration",
        "base_effort": 311.52,
        "min": 1,
        "max": 6,
        "dependencies": [
            36
        ],
        "resource": "SDU SolArch (JS6) - DevOps SME"
    },
    {
        "id": 38,
        "task_name": "Software upgrade - CBEV",
        "base_effort": 429.62823529411776,
        "min": 1,
        "max": 5,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 39,
        "task_name": "Application - Security Hardening",
        "base_effort": 184.95999999999998,
        "min": 1,
        "max": 5,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 40,
        "task_name": "Security Risk Assessment (Detailed-RA) – SI Project Scoping",
        "base_effort": 100,
        "min": 1,
        "max": 5,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Security Master"
    },
    {
        "id": 41,
        "task_name": "Privacy Impact Assessment (PIA)",
        "base_effort": 40,
        "min": 1,
        "max": 5,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Security Master"
    },
    {
        "id": 42,
        "task_name": "Secure Hardening",
        "base_effort": 160,
        "min": 1,
        "max": 5,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 43,
        "task_name": "Vulnerability Analysis (VA)",
        "base_effort": 131.4,
        "min": 1,
        "max": 5,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Security Master"
    },
    {
        "id": 44,
        "task_name": "Agile Tool Setup (CDD or GITOPS)",
        "base_effort": 100,
        "min": 1,
        "max": 5,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 45,
        "task_name": "Test Automation – Tool installation and framework setup (ESSVT or other)",
        "base_effort": 23.76,
        "min": 1,
        "max": 6,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 46,
        "task_name": "CBEV: Customer Questionnaire",
        "base_effort": 272,
        "min": 1,
        "max": 6,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 47,
        "task_name": "Lexie input file generation",
        "base_effort": 64,
        "min": 1,
        "max": 6,
        "dependencies": [
            37
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 48,
        "task_name": "Initial Application installation - Manual (NELS server optional)",
        "base_effort": 352.23529411764713,
        "min": 1,
        "max": 6,
        "dependencies": [
            47
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 49,
        "task_name": "Solution Monitoring Setup and Configuration",
        "base_effort": 88,
        "min": 1,
        "max": 6,
        "dependencies": [
            48
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 50,
        "task_name": "Integration with IAM",
        "base_effort": 200,
        "min": 1,
        "max": 6,
        "dependencies": [
            48
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 51,
        "task_name": "User migration for integration with IAM",
        "base_effort": 80,
        "min": 1,
        "max": 6,
        "dependencies": [
            50
        ],
        "resource": "SDU SolArch (JS6) - Infra or SW Integration SA"
    },
    {
        "id": 52,
        "task_name": "CBEV Solution Architects",
        "base_effort": 1360,
        "min": 1,
        "max": 3,
        "dependencies": [
            19
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 53,
        "task_name": "CHA Configuration and Build",
        "base_effort": 308.88,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 54,
        "task_name": "CHA Configuration and Build - Interfaces",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 55,
        "task_name": "D3 Miscellaneous effort - CHA",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 56,
        "task_name": "CIL Configuration and Build",
        "base_effort": 56,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 57,
        "task_name": "D3 Miscellaneous effort - CIL",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 58,
        "task_name": "MSV Configuration & Build (Virtualized)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 59,
        "task_name": "BAM (Business Application Management (Containerized)",
        "base_effort": 80,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 60,
        "task_name": "D3 Miscellaneous effort - MSV",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 61,
        "task_name": "INV Configuration and Build",
        "base_effort": 4,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 62,
        "task_name": "INV External Interface",
        "base_effort": 80,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 63,
        "task_name": "FIN Configuration and Build",
        "base_effort": 40,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 64,
        "task_name": "FIN External Interface",
        "base_effort": 176,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 65,
        "task_name": "COBA Configuration and Build",
        "base_effort": 56,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 66,
        "task_name": "COBA External Interface Configuration",
        "base_effort": 36,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 67,
        "task_name": "D3 Miscellaneous effort - COBA",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 68,
        "task_name": "EDM Configuration and Build",
        "base_effort": 224,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 69,
        "task_name": "EPS Configuration and Build",
        "base_effort": 144,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 70,
        "task_name": "D3 Miscellaneous effort - EDM or EPS",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 71,
        "task_name": "NTF Configuration and Build (Templates)",
        "base_effort": 103.99999999999999,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 72,
        "task_name": "NTF Templates (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 73,
        "task_name": "NTF External Interface",
        "base_effort": 8,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 74,
        "task_name": "CPM Configuration and Build",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 75,
        "task_name": "D3 Miscellaneous effort - CPM",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 76,
        "task_name": "AEL Configuration and Build (Use case)",
        "base_effort": 1200,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 77,
        "task_name": "AEL External Interface Configuration (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 78,
        "task_name": "AEL - AEP - Workflow (CAMEL or CAMUNDA) (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 79,
        "task_name": "BSS API Exposure (BAE) DV (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 80,
        "task_name": "D3 Miscellaneous effort - AEL",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 81,
        "task_name": "Business Configuration (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 82,
        "task_name": "RMCA - Product Bundles (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 83,
        "task_name": "RMCA - Product Offers (Similar Pos)",
        "base_effort": 330.28000000000003,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 84,
        "task_name": "RMCA - Product Offers (Unique Pos)",
        "base_effort": 576,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 85,
        "task_name": "RMCA - Interface",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 86,
        "task_name": "RMCA - Policy counter configuration (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 87,
        "task_name": "RMCA - CFS (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 88,
        "task_name": "RMCA - RFS (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 89,
        "task_name": "RMCA - Logical and Physical Resources (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 90,
        "task_name": "RMCA - Rules (Not in use) (Not in use)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 91,
        "task_name": "D3 Miscellaneous effort - RMCA",
        "base_effort": 320,
        "min": 1,
        "max": 6,
        "dependencies": [
            51
        ],
        "resource": "SA/SD SolArch (JS6) - CBEV Software Developer"
    },
    {
        "id": 92,
        "task_name": "ST Triage and Management",
        "base_effort": 400,
        "min": 1,
        "max": 3,
        "dependencies": [
            76
        ],
        "resource": "SDU CPM (JS7) - Rollout Manager"
    },
    {
        "id": 93,
        "task_name": "ST Functional",
        "base_effort": 838.5558635294119,
        "min": 1,
        "max": 6,
        "dependencies": [
            76
        ],
        "resource": "SDU IntegEngin (JS4) - ST Test Engineer"
    },
    {
        "id": 94,
        "task_name": "ST Non-functional (Security, Performance, Penetration Testing)",
        "base_effort": 0,
        "min": 1,
        "max": 3,
        "dependencies": [
            76
        ],
        "resource": "SDU IntegEngin (JS6) - CBEV Solution Integrator"
    },
    {
        "id": 95,
        "task_name": "ST Bug Fixing",
        "base_effort": 838.5558635294119,
        "min": 1,
        "max": 6,
        "dependencies": [
            76
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 96,
        "task_name": "Triage and Management",
        "base_effort": 320,
        "min": 1,
        "max": 3,
        "dependencies": [
            76
        ],
        "resource": "SDU CPM (JS7) - Rollout Manager"
    },
    {
        "id": 97,
        "task_name": "Test Support & Bug fixing - CBEV",
        "base_effort": 749.2560000000002,
        "min": 1,
        "max": 3,
        "dependencies": [
            76
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 98,
        "task_name": "Data Migration PM",
        "base_effort": 2200,
        "min": 1,
        "max": 3,
        "dependencies": [
            19
        ],
        "resource": "SDU SolArch (JS5) - RME or RTE"
    },
    {
        "id": 99,
        "task_name": "Data Migration Requirement Gathering",
        "base_effort": 300,
        "min": 1,
        "max": 3,
        "dependencies": [
            19
        ],
        "resource": "SDU CPM (JS7) - DM CPM"
    },
    {
        "id": 100,
        "task_name": "Data Migration Build - Tool development",
        "base_effort": 9.600000000000009,
        "min": 1,
        "max": 6,
        "dependencies": [
            400
        ],
        "resource": "SA/SD SolArch (JS7) - DM Lead / E2E SA"
    },
    {
        "id": 101,
        "task_name": "Data Migration Build - CBEV",
        "base_effort": 6240,
        "min": 1,
        "max": 6,
        "dependencies": [
            400
        ],
        "resource": "SA/SD SolArch (JS7) - DM Lead / E2E SA"
    },
    {
        "id": 102,
        "task_name": "Data Migration SIT",
        "base_effort": 320,
        "min": 1,
        "max": 6,
        "dependencies": [
            408
        ],
        "resource": "SA/SD SolArch (JS7) - DM Lead / E2E SA"
    },
    {
        "id": 103,
        "task_name": "Data Migration UAT",
        "base_effort": 320,
        "min": 1,
        "max": 6,
        "dependencies": [
            102
        ],
        "resource": "SA/SD SolArch (JS7) - DM Lead / E2E SA"
    },
    {
        "id": 104,
        "task_name": "Data Migration Dry Run",
        "base_effort": 320,
        "min": 1,
        "max": 6,
        "dependencies": [
            103
        ],
        "resource": "SA/SD SolArch (JS7) - DM Lead / E2E SA"
    },
    {
        "id": 105,
        "task_name": "Data Migration Roll Out",
        "base_effort": 320,
        "min": 1,
        "max": 6,
        "dependencies": [
            104
        ],
        "resource": "SA/SD SolArch (JS7) - DM Lead / E2E SA"
    },
    {
        "id": 106,
        "task_name": "Data Migration PPS",
        "base_effort": 320,
        "min": 1,
        "max": 6,
        "dependencies": [
            105
        ],
        "resource": "SA/SD SolArch (JS7) - DM Lead / E2E SA"
    },
    {
        "id": 107,
        "task_name": "E2E Test Support",
        "base_effort": 288.2484561728751,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "MA TestMgr (JS6) - Test lead"
    },
    {
        "id": 108,
        "task_name": "Test Automation - Scripting - API",
        "base_effort": 129.92940000000004,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 109,
        "task_name": "Test Automation - Scripting - DB",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 110,
        "task_name": "Test Automation - Scripting - SSH",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 111,
        "task_name": "Test Automation - Peer review of Script and Feedback Incorporation",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 112,
        "task_name": "Test Automation - Integration with DevOps",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 113,
        "task_name": "Test Automation - Execution and reporting - GUI",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 114,
        "task_name": "Test Automation - Execution and reporting - API",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 115,
        "task_name": "Test Automation - Execution result analysis-GUI",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 116,
        "task_name": "Test Automation - Execution result analysis-API",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 117,
        "task_name": "Test Automation – Maintenance",
        "base_effort": 640,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SDU ITEngTest (JS4) - SIT Manual Tester"
    },
    {
        "id": 118,
        "task_name": "Internal GeoRedundancy execution",
        "base_effort": 26.880000000000003,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 119,
        "task_name": "HA or Failover Testing",
        "base_effort": 186.88,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 120,
        "task_name": "Backup-Restore Testing",
        "base_effort": 121.60000000000001,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 121,
        "task_name": "Security Testing",
        "base_effort": 245.76,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 122,
        "task_name": "Alarm Testing",
        "base_effort": 90.88,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 123,
        "task_name": "Monitoring & Logging Testing",
        "base_effort": 75.52000000000001,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 124,
        "task_name": "User Management",
        "base_effort": 140.8,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 125,
        "task_name": "Geo-Red Testing",
        "base_effort": 211.20000000000002,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD TechSME (JS6) - NFR SA"
    },
    {
        "id": 126,
        "task_name": "PT - Understanding Application or Architecture",
        "base_effort": 38.400000000000006,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 127,
        "task_name": "PT - POC (Optional)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 128,
        "task_name": "PT - Environment Setup and Tool Setup Validation",
        "base_effort": 184,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 129,
        "task_name": "PT - Production Statistics and Understanding KPI or SLA and Create Test Plan",
        "base_effort": 255.68,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 130,
        "task_name": "PT - Functional flows understanding of PT Scenarios",
        "base_effort": 276.08000000000004,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 131,
        "task_name": "PT - Test Data Identification and Validation",
        "base_effort": 925.5999999999999,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 132,
        "task_name": "PT - Scripting (GUI or API)",
        "base_effort": 1067.2,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 133,
        "task_name": "PT - Load Execution and Monitoring",
        "base_effort": 599.76,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 134,
        "task_name": "PT - Stress Execution and Monitoring",
        "base_effort": 212.16000000000003,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 135,
        "task_name": "PT - Endurance Execution and Monitoring",
        "base_effort": 19.200000000000003,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 136,
        "task_name": "PT - Result Analysis and Defect Reporting",
        "base_effort": 76,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 137,
        "task_name": "PT - Final Analysis and Report Submission",
        "base_effort": 204,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 138,
        "task_name": "PT - Tuning recommendations (Optional)",
        "base_effort": 104.4,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 139,
        "task_name": "PT - Retest (For every test as needed)",
        "base_effort": 64.80000000000001,
        "min": 1,
        "max": 6,
        "dependencies": [
            106
        ],
        "resource": "SA/SD SolArch (JS7) - Performance SA"
    },
    {
        "id": 140,
        "task_name": "User Acceptance Testing-Testing",
        "base_effort": 807.0956772840501,
        "min": 1,
        "max": 6,
        "dependencies": [
            138
        ],
        "resource": "MA TestMgr (JS6) - UAT Test Manager"
    },
    {
        "id": 141,
        "task_name": "Release or Configuration Manager",
        "base_effort": 520,
        "min": 1,
        "max": 6,
        "dependencies": [
            140
        ],
        "resource": "SDU ProgMgr (JS8) - Factory / Delivery Lead"
    },
    {
        "id": 142,
        "task_name": "Rollout Manager",
        "base_effort": 120,
        "min": 1,
        "max": 6,
        "dependencies": [
            140
        ],
        "resource": "MA CPM (JS7) - PMO"
    },
    {
        "id": 143,
        "task_name": "Continuous or Manual Deployment (Build and Testing period)",
        "base_effort": 0,
        "min": 1,
        "max": 6,
        "dependencies": [
            140
        ],
        "resource": "SDU ITEngTest (JS5) - ST Automation Test Engineer"
    },
    {
        "id": 144,
        "task_name": "Solution Cut-Over Planning and execution",
        "base_effort": 188.88419294117642,
        "min": 1,
        "max": 6,
        "dependencies": [
            140
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 145,
        "task_name": "Handover to Customer Operations Organization",
        "base_effort": 120,
        "min": 1,
        "max": 6,
        "dependencies": [
            140
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 146,
        "task_name": "Post Production Support (bug fixing+deployment)",
        "base_effort": 720,
        "min": 1,
        "max": 6,
        "dependencies": [
            145
        ],
        "resource": "SDU SwDev (JS6) - CBEV Software Developer"
    },
    {
        "id": 147,
        "task_name": "LCM - CBEV",
        "base_effort": 1274.7568174380528,
        "min": 1,
        "max": 6,
        "dependencies": [
            145
        ],
        "resource": "SDU ITSysExp (JS6) - LCM Support Team"
    }
]
    resource_pool = {
    "SA/SD SolArch (JS8) - Solution Architect - DA": 1000,
    "SDU IntegEngin (JS6) - CBEV Solution Integrator": 1000,
    "SDU SolArch (JS6) - DevOps Tool Admin": 1000,
    "SDU SolArch (JS6) - Performance Lead": 1000,
    "SDU CPM (JS7) - DM CPM": 1000,
    "MA TestMgr (JS6) - Test lead": 1000,
    "SDU ITSysExp (JS5) - SW SME": 1000,
    "SDU IntegEngin (JS6) - CAF IE": 1000,
    "SDU SwDev (JS6) - CBEV Software Developer": 1000,
    "SA/SD TechSME (JS6) - NFR SA": 1000,
    "SDU SolArch (JS6) - Infra or SW Integration SA": 1000,
    "SDU SolArch (JS6) - DevOps SME": 1000,
    "SDU SolArch (JS6) - Security Master": 1000,
    "SA/SD SolArch (JS6) - CBEV Software Developer": 1000,
    "SDU CPM (JS7) - Rollout Manager": 1000,
    "SDU IntegEngin (JS4) - ST Test Engineer": 1000,
    "SDU SolArch (JS5) - RME or RTE": 1000,
    "SA/SD SolArch (JS7) - DM Lead / E2E SA": 1000,
    "SDU ITEngTest (JS4) - SIT Manual Tester": 1000,
    "SA/SD SolArch (JS7) - Performance SA": 1000,
    "MA TestMgr (JS6) - UAT Test Manager": 1000,
    "SDU ProgMgr (JS8) - Factory / Delivery Lead": 1000,
    "MA CPM (JS7) - PMO": 1000,
    "SDU ITEngTest (JS5) - ST Automation Test Engineer": 1000,
    "SDU ITSysExp (JS6) - LCM Support Team": 1000
}
    resource_cost = {
    "SA/SD SolArch (JS8) - Solution Architect - DA": 118.97,
    "SDU IntegEngin (JS6) - CBEV Solution Integrator": 23.82,
    "SDU SolArch (JS6) - DevOps Tool Admin": 23.82,
    "SDU SolArch (JS6) - Performance Lead": 23.82,
    "SDU CPM (JS7) - DM CPM": 31.86,
    "MA TestMgr (JS6) - Test lead": 83.68,
    "SDU ITSysExp (JS5) - SW SME": 19.26,
    "SDU IntegEngin (JS6) - CAF IE": 23.82,
    "SDU SwDev (JS6) - CBEV Software Developer": 19.26,
    "SA/SD TechSME (JS6) - NFR SA": 92.78,
    "SDU SolArch (JS6) - Infra or SW Integration SA": 23.82,
    "SDU SolArch (JS6) - DevOps SME": 23.82,
    "SDU SolArch (JS6) - Security Master": 23.82,
    "SA/SD SolArch (JS6) - CBEV Software Developer": 37.1,
    "SDU CPM (JS7) - Rollout Manager": 31.86,
    "SDU IntegEngin (JS4) - ST Test Engineer": 14.66,
    "SDU SolArch (JS5) - RME or RTE": 23.82,
    "SA/SD SolArch (JS7) - DM Lead / E2E SA": 93.21,
    "SDU ITEngTest (JS4) - SIT Manual Tester": 14.66,
    "SA/SD SolArch (JS7) - Performance SA": 93.21,
    "MA TestMgr (JS6) - UAT Test Manager": 83.68,
    "SDU ProgMgr (JS8) - Factory / Delivery Lead": 44.43,
    "MA CPM (JS7) - PMO": 93.21,
    "SDU ITEngTest (JS5) - ST Automation Test Engineer": 14.66,
    "SDU ITSysExp (JS6) - LCM Support Team": 19.26
}
    
    return tasks, resource_pool, resource_cost