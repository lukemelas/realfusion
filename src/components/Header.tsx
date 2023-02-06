import { Heading, Wrap, Box, Container, Link as ChakraLink } from '@chakra-ui/react'
import NextLink from "next/link"

import { title, institutions, authors } from 'data'


export const Title = () => (
  <Heading fontWeight="light" fontSize="3xl" pt="3rem" maxW="60rem" textAlign="center">{title}</Heading>
)


export const Authors = () => (
  <Container maxW="60rem">
    <Wrap justify="center" pt="1rem" fontSize="xl" key="authors">
      {
        authors.map((author) =>
          <Box key={author.name} pl="1rem" pr="1rem">
            <NextLink href={author.url} passHref={true}>
              {author.name}
            </NextLink>
            <sup> {author.institutions.toString()}</sup>
          </Box>
        )
      }
    </Wrap>
    <Wrap justify="center" pt="1rem" key="institutions">
      {
        Object.entries(institutions).map(tuple =>
          <Box key={tuple[0]}>
            <sup>{tuple[0]}  </sup>
            {tuple[1]}
          </Box>
        )
      }
    </Wrap>
  </Container>
)