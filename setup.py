from setuptools import find_packages, setup

import os
import sys


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


VERSION_FILE = os.path.abspath('computeEvaltool/version.py')


def get_version():
    with open(VERSION_FILE, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), VERSION_FILE, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            relative_base = os.path.dirname(fname)
            absolute_target = os.path.join(relative_base, target)
            for info in parse_require_file(absolute_target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('http') and not line.startswith('git+http'):
                    print('skip http requirements %s' % line)
                    continue
                # Handle git+https:// dependencies
                if line.startswith('git+http'):
                    # Extract package name from git URL
                    if '#egg=' in line:
                        package_name = line.split('#egg=')[1]
                        info = {'line': line, 'package': package_name, 'is_git_dep': True, 'git_url': line}
                        yield info
                    else:
                        print('skip git requirements without egg name: %s' % line)
                    continue
                if line and not line.startswith('#') and not line.startswith('--'):
                    for info in parse_line(line):
                        yield info
                elif line and line.startswith('--find-links'):
                    eles = line.split()
                    for e in eles:
                        e = e.strip()
                        if 'http' in e:
                            info = dict(dependency_links=e)
                            yield info

    def gen_packages_items():
        items = []
        deps_link = []
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                if 'dependency_links' not in info:
                    # Handle git dependencies
                    if info.get('is_git_dep'):
                        # For git dependencies, add the package name, not the full URL
                        # The git URL should go to dependency_links
                        items.append(info['package'])
                        deps_link.append(info['git_url'])
                    else:
                        parts = [info['package']]
                        if with_version and 'version' in info:
                            parts.extend(info['version'])
                        if not sys.version.startswith('3.4'):
                            # apparently package_deps are broken in 3.4
                            platform_deps = info.get('platform_deps')
                            if platform_deps is not None:
                                parts.append(';' + platform_deps)
                        item = ''.join(parts)
                        items.append(item)
                else:
                    deps_link.append(info['dependency_links'])
        return items, deps_link

    return gen_packages_items()


if __name__ == '__main__':
    print('Usage: python3 setup.py bdist_wheel or pip3 install .[opencompass] for test')

    install_requires, deps_link = parse_requirements('requirements/framework.txt')

    extra_requires = {}
    all_requires = []
    extra_requires['perf'], _ = parse_requirements('requirements/perf.txt')


    all_requires.extend(install_requires)
    all_requires.extend(extra_requires['perf'])
    extra_requires['all'] = all_requires

    setup(
        name='computeEvaltool',
        version=get_version(),
        author='Suanli Group',
        author_email='contact@cs.ac.cn',
        keywords='python,llm,computility evaluate tool,',
        description='COMPUTEEVALTOOL: A LightWeight Computility Evalution Framework',
        long_description=readme(),
        url='https://git.cs.ac.cn/hfzh/computeevaltool',
        include_package_data=True,
        package_data={
            'computeEvaltool': [
                'registry/tasks/*.yaml',
                'benchmarks/bbh/cot_prompts/*.txt',
                'third_party/longbench_write/resources/*',
            ]
        },
        packages=find_packages(exclude=('configs', 'demo')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
        ],
        python_requires='>=3.9',
        zip_safe=False,
        install_requires=install_requires,
        entry_points={'console_scripts': ['computeEvaltool=computeEvaltool.cli.cli:run_cmd']},
        dependency_links=deps_link,
        extras_require=extra_requires,
        license='Apache License 2.0',
    )