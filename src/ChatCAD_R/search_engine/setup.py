from setuptools import Extension, setup, find_packages

setup(name='chatcad_search_engine',
        version='0.1',
        description='Search engine for chatcad_plus',
        long_description="See Readme.md on github for more details.",
        author='CH2',
        python_requires='>=3.5',
        license='Apache 2.0',
        packages=find_packages(),
        install_requires=[
            'scipy',
            'scikit-learn',
            'pandas',
            'jieba'
        ],
        zip_safe=False,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        scripts=[
            # 'bin/TractSegpp'
        ]
    )
