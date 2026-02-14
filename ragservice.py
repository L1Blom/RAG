"""
New RAG service using refactored modules.

This file serves as a migration bridge.  Run it exactly like the
original ``ragservice.py``::

    python ragservice_new.py <project>

Once you've verified that the new implementation works correctly,
rename this file to ``ragservice.py`` (and archive the old one).
"""

from rag.app import main

if __name__ == '__main__':
    main()
