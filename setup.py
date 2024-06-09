from setuptools import setup, find_packages

setup(
    name='SentenceFromSpace',
    version='0.1.0',
    description='Sentence generation from latent space using VAE and LSTM.',
    author='Itay Etelis, Yair Hadas',
    url='https://github.com/yourusername/SentenceFromSpace',
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'tqdm>=4.61.0',
        'PyYAML>=5.4.1',
        'tensorboard>=2.5.0'
    ],
    entry_points={
        'console_scripts': [
            'train_sentence_model=training.train:main',
            'predict_sentence_model=inference.predict:main'
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
