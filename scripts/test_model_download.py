"""No-network tests for interrupted model-download retries."""
import unittest
from unittest.mock import Mock
from requests.exceptions import ChunkedEncodingError
from huggingface_hub.utils import LocalEntryNotFoundError, EntryNotFoundError
from prepare_nli_models import download_file


class DownloadTests(unittest.TestCase):
    def test_cached_file_does_not_contact_network(self):
        download = Mock(return_value='cached-file')
        self.assertEqual(download_file('repo', 'file', download=download), 'cached-file')
        download.assert_called_once_with(repo_id='repo', filename='file', local_files_only=True)

    def test_interrupted_transfer_retries_without_forcing_restart(self):
        download = Mock(side_effect=[LocalEntryNotFoundError('missing'), ChunkedEncodingError('interrupted'), 'complete-file'])
        sleep = Mock()
        self.assertEqual(download_file('repo', 'file', download=download, sleep=sleep), 'complete-file')
        self.assertEqual(download.call_count, 3)
        self.assertTrue(all(call.kwargs.get('force_download') is False for call in download.call_args_list[1:]))
        sleep.assert_called_once_with(5)

    def test_missing_remote_file_is_not_retried(self):
        download = Mock(side_effect=[LocalEntryNotFoundError('missing'), EntryNotFoundError('404')])
        with self.assertRaises(EntryNotFoundError):
            download_file('repo', 'file', download=download, sleep=Mock())
        self.assertEqual(download.call_count, 2)


if __name__ == '__main__':
    unittest.main()
